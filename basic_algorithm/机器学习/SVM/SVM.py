# ==============================
# SVM完整实验代码
# 环境：scikit-learn==1.9.0
# 功能：
# 1. 二维带噪声数据集：硬间隔SVM、软间隔SVM左右子图同框对比
# 2. 错分样本：大红色空心外圈标记；支持向量黑色空心圈，重叠时双层圆圈共存
# 3. 三维沿Z轴可分数据集，marching_cubes绘制平滑3D分隔超平面
# 4. 自动打印：样本到超平面距离、支持向量坐标、各类支持向量数量、支持向量索引
# ==============================

# 导入数值计算库numpy，用于数组生成、矩阵运算
import numpy as np
# 导入绘图库matplotlib，实现二维、三维可视化
import matplotlib.pyplot as plt
# 导入3D绘图工具，用于创建三维坐标轴
from mpl_toolkits.mplot3d import Axes3D
# 导入sklearn SVM分类器SVC，底层基于libsvm实现支持向量机
from sklearn.svm import SVC
# 导入标准化工具，SVM对特征尺度极度敏感，必须标准化
from sklearn.preprocessing import StandardScaler
# 导入准确率评估指标，计算模型预测正确率
from sklearn.metrics import accuracy_score
# 导入三维等值面提取算法，用于绘制平滑连续的3D决策曲面，消除之前柱状畸形
from skimage.measure import marching_cubes

# 设置全局字体：支持中文显示，优先使用黑体
plt.rcParams["font.family"] = ["SimHei"]
# 设置负号正常显示，避免坐标轴负刻度显示方框乱码
plt.rcParams["axes.unicode_minus"] = False
# 设置随机种子，固定随机数输出，保证每次运行数据集完全一致，结果可复现
np.random.seed(42)

# ====================== 工具函数1：绘制二维软硬间隔左右对比子图 ======================
def plot_svm_pair(clf_hard, clf_soft, X, y):
    """
    绘制左右并排子图：左侧硬间隔SVM，右侧软间隔SVM
    :param clf_hard: 训练完成的硬间隔SVM模型
    :param clf_soft: 训练完成的软间隔SVM模型
    :param X: 标准化后的二维特征数据集
    :param y: 数据集真实标签
    """
    # 创建画布，1行2列子图，画布尺寸宽14英寸，高6英寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # 获取x轴坐标上下限，向外扩充0.8，留出绘图边距
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    # 获取y轴坐标上下限，向外扩充0.8，留出绘图边距
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    # 在x范围内生成200个等间距采样点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # 将网格坐标展平，拼接成N×2数组，用于模型预测网格所有点
    grid = np.c_[xx.ravel(), yy.ravel()]

    # -------------------- 左图：硬间隔 SVM --------------------
    # 计算网格每个点到分隔超平面的距离（决策函数值）
    dec_hard = clf_hard.decision_function(grid).reshape(xx.shape)
    # 填充等高区域，展示分类区域，alpha透明度0.25，配色coolwarm冷暖色
    ax1.contourf(xx, yy, clf_hard.predict(grid).reshape(xx.shape), alpha=0.25, cmap="coolwarm")
    # 绘制三条线：-1间隔边界(灰色虚线)、0决策边界(黑色实线)、+1间隔边界(灰色虚线)
    ax1.contour(xx, yy, dec_hard, levels=[-1, 0, 1],
                colors=["gray", "black", "gray"],
                linestyles=["--", "-", "--"], linewidths=[1.5, 2, 1.5])
    # 使用硬间隔模型预测所有样本标签
    y_pred_hard = clf_hard.predict(X)
    # 绘制所有样本点，颜色为预测标签，黑色边框，大小60
    ax1.scatter(X[:, 0], X[:, 1], c=y_pred_hard, cmap="coolwarm", edgecolors="k", s=60)
    # 绘制支持向量：黑色空心圆圈，大小220，线宽2，图例标签
    ax1.scatter(clf_hard.support_vectors_[:, 0], clf_hard.support_vectors_[:, 1],
                s=220, facecolors="none", edgecolor="black", lw=2, label="支持向量")
    # 设置子图标题
    ax1.set_title("硬间隔SVM | Linear C=1e9\n强制区分噪声，间隔狭窄，易过拟合")
    # 显示图例
    ax1.legend()

    # -------------------- 右图：软间隔 SVM --------------------
    # 计算网格每个点到分隔超平面的距离
    dec_soft = clf_soft.decision_function(grid).reshape(xx.shape)
    # 填充分类区域底色
    ax2.contourf(xx, yy, clf_soft.predict(grid).reshape(xx.shape), alpha=0.25, cmap="coolwarm")
    # 绘制最大间隔边界线与决策分界线
    ax2.contour(xx, yy, dec_soft, levels=[-1, 0, 1],
                colors=["gray", "black", "gray"],
                linestyles=["--", "-", "--"], linewidths=[1.5, 2, 1.5])
    # 软间隔模型预测样本标签
    y_pred_soft = clf_soft.predict(X)
    # 绘制全部样本，颜色使用【预测类别】颜色
    ax2.scatter(X[:, 0], X[:, 1], c=y_pred_soft, cmap="coolwarm", edgecolors="k", s=60)
    # 绘制软间隔支持向量，黑色空心圈
    ax2.scatter(clf_soft.support_vectors_[:, 0], clf_soft.support_vectors_[:, 1],
                s=220, facecolors="none", edgecolor="black", lw=2, label="支持向量")
    # 生成布尔掩码：预测标签 != 真实标签，筛选出错分样本
    wrong_mask = (y_pred_soft != y)
    # 判断是否存在错分样本
    if np.any(wrong_mask):
        # 错分样本绘制更大的红色空心外圈，线宽2.5；支持向量会内层黑圈、外层红圈双层叠加
        ax2.scatter(X[wrong_mask, 0], X[wrong_mask, 1],
                    s=350, facecolors="none", edgecolor="red", lw=2.5, label="错分样本")
    # 设置右侧子图标题
    ax2.set_title("软间隔SVM | Linear C=0.2\n容忍噪声错分，间隔更宽，鲁棒性更强")
    # 显示图例
    ax2.legend()
    # 自动调整子图间距，防止标题、标签重叠
    plt.tight_layout()
    # 弹出画布窗口展示图像
    plt.show()

# ====================== 工具函数2：3D平滑决策曲面绘制 ======================
def plot_3d_decision_surface(clf, X, y, title):
    """
    绘制三维散点 + SVM决策等值曲面
    :param clf: 训练完成的3D SVM模型
    :param X: 标准化后的三维特征
    :param y: 真实标签
    :param title: 图像标题文本
    """
    # 创建画布，尺寸10×8英寸
    fig = plt.figure(figsize=(10, 8))
    # 添加3D坐标轴子图
    ax = fig.add_subplot(111, projection="3d")
    # 绘制三维样本散点，冷暖配色，黑色描边，大小50
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="coolwarm", s=50, edgecolor="k")

    # 获取X1轴范围，向外扩充边距
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    # 获取X2轴范围，向外扩充边距
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    # 获取X3(Z)轴范围，向外扩充边距
    z_min, z_max = X[:,2].min()-0.5, X[:,2].max()+0.5
    # 网格分辨率，数值越大曲面越精细，渲染越慢
    grid_res = 40
    # 生成三维网格采样点
    xx, yy, zz = np.mgrid[x_min:x_max:grid_res*1j,
                          y_min:y_max:grid_res*1j,
                          z_min:z_max:grid_res*1j]
    # 将三维网格坐标展平拼接为 N×3 矩阵，送入模型预测
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    # 计算网格所有点到分隔超平面距离
    dec = clf.decision_function(grid_points).reshape(xx.shape)

    # marching_cubes：提取等值面，level=0代表decision_function=0的决策边界曲面
    verts, faces, _, _ = marching_cubes(dec, level=0)
    # 将算法内部归一化顶点坐标映射回原始坐标轴范围
    verts[:,0] = x_min + (x_max-x_min) * verts[:,0]/(grid_res-1)
    verts[:,1] = y_min + (y_max-y_min) * verts[:,1]/(grid_res-1)
    verts[:,2] = z_min + (z_max-z_min) * verts[:,2]/(grid_res-1)
    # 绘制三角面片构成的决策曲面，透明度0.4，浅蓝色
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, alpha=0.4, color="deepskyblue")

    # 设置坐标轴标签
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3 (Z轴)")
    # 设置图像标题
    ax.set_title(title)
    # 展示画布
    plt.show()

# ====================== 工具函数3：统一打印SVM模型详细信息 ======================
def print_svm_info(model, X_train, dataset_name):
    """
    批量打印SVM关键信息：距离、支持向量、各类支持向量数量、支持向量索引
    :param model: 训练完成的SVC模型
    :param X_train: 训练特征集
    :param dataset_name: 数据集名称，用于打印区分
    """
    print(f"\n==========【{dataset_name} SVM模型信息】==========")
    # decision_function：输出每个样本到分隔超平面的有符号距离
    dists = model.decision_function(X_train)
    # 打印前10条距离，保留4位小数，避免输出过长刷屏
    print(f"训练样本到分隔超平面距离（前10条）：\n{dists[:10].round(4)}")

    # .support_vectors_ 属性：返回所有支持向量特征坐标
    sv = model.support_vectors_
    print(f"\n支持向量坐标：\n{sv.round(4)}")

    # .n_support_：数组，依次保存类别0、类别1各自的支持向量数量
    n_sv_per_class = model.n_support_
    print(f"\n各类别支持向量数量：类别0={n_sv_per_class[0]}个，类别1={n_sv_per_class[1]}个")

    # .support_：支持向量在训练集X_train中的原始索引
    sv_idx = model.support_
    print(f"\n支持向量在训练集索引：{sv_idx.tolist()}")
    # 分割线，区分不同模型输出
    print("="*60)

# =====================================================================
# 实验1：二维带噪声数据集｜硬间隔 & 软间隔同框对比
# =====================================================================
print("========== 实验1：二维带噪声线性可分数据 ==========")
# 生成类别0基础样本：均值[-2,0]，标准差0.6，共50个二维样本
X1 = np.random.normal(loc=[-2, 0], scale=0.6, size=(50, 2))
# 生成类别1基础样本：均值[2,0]，标准差0.6，共50个二维样本
X2 = np.random.normal(loc=[2, 0], scale=0.6, size=(50, 2))
# 手动构造2个跨区域噪声异常点，制造干扰
noise_points = np.array([[0.2, -0.4], [-0.3, 0.6]])
# 垂直拼接所有特征：类别0 + 类别1 + 噪声点
X = np.vstack([X1, X2, noise_points])
# 拼接标签：50个0 + 50个1 + 噪声标签[0,1]
y = np.hstack([np.zeros(50), np.ones(50), np.array([0, 1])])

# 创建标准化转换器
scaler = StandardScaler()
# fit_transform：先统计训练集均值方差，再标准化特征
X_sc = scaler.fit_transform(X)

# 构建硬间隔SVM：线性核，C极大(1e9)，对违反间隔样本施加极高惩罚
# kernel="linear"：线性核；C：惩罚系数，C越大越接近硬间隔
svm_hard = SVC(kernel="linear", C=1e9)
# 使用标准化数据训练模型
svm_hard.fit(X_sc, y)
# 模型预测训练集全部样本
pred_hard = svm_hard.predict(X_sc)
# 计算并打印训练集准确率
print(f"硬间隔准确率: {accuracy_score(y, pred_hard):.3f}")
# 调用函数打印硬间隔模型完整信息
print_svm_info(svm_hard, X_sc, "硬间隔线性SVM")

# 构建软间隔SVM：线性核，C=0.2，惩罚较小，允许少量样本错分
svm_soft = SVC(kernel="linear", C=0.2)
# 训练软间隔模型
svm_soft.fit(X_sc, y)
# 预测训练集样本
pred_soft = svm_soft.predict(X_sc)
# 打印软间隔准确率
print(f"软间隔准确率: {accuracy_score(y, pred_soft):.3f}")
# 打印软间隔模型完整信息
print_svm_info(svm_soft, X_sc, "软间隔线性SVM")

# 调用绘图函数，绘制左右对比图
plot_svm_pair(svm_hard, svm_soft, X_sc, y)

# =====================================================================
# 实验2：三维沿Z轴可分数据集【拉大两类距离，样本不拥挤重叠】
# =====================================================================
print("\n========== 实验2：三维沿Z轴分隔数据（两类拉开距离） ==========")
# 总样本数量
n = 180
# X1特征：在[-2,2]区间均匀随机采样
X1 = np.random.uniform(-2, 2, n)
# X2特征：在[-2,2]区间均匀随机采样
X2 = np.random.uniform(-2, 2, n)

# Z轴类别0：均值-1.3，标准差0.4，样本数量一半
X3_class0 = np.random.normal(loc=-1.3, scale=0.4, size=n//2)
# Z轴类别1：均值1.3，标准差0.4，样本数量一半；两类中心差距大，空隙明显
X3_class1 = np.random.normal(loc=1.3, scale=0.4, size=n//2)
# 拼接Z轴特征
X3 = np.hstack([X3_class0, X3_class1])
# 拼接标签：前一半0，后一半1
y = np.hstack([np.zeros(n//2), np.ones(n//2)])

# 将X1,X2,X3合并为三维特征矩阵 N×3
X_3d = np.column_stack([X1, X2, X3])

# 创建标准化工具
scaler3d = StandardScaler()
# 标准化三维特征
X3d_sc = scaler3d.fit_transform(X_3d)

# 三维线性核SVM，C=1.0
svm_linear_3d = SVC(kernel="linear", C=1.0)
# 训练模型
svm_linear_3d.fit(X3d_sc, y)
# 预测样本标签
pred_3d_lin = svm_linear_3d.predict(X3d_sc)
# 打印准确率
print(f"三维Linear-SVM准确率: {accuracy_score(y, pred_3d_lin):.3f}")
# 打印模型信息
print_svm_info(svm_linear_3d, X3d_sc, "三维线性SVM")
# 绘制3D散点+分隔平面
plot_3d_decision_surface(svm_linear_3d, X3d_sc, y,
    "三维数据【沿Z轴可分】| Linear SVM\n蓝色平面 = 线性分隔超平面")

# 三维RBF高斯核SVM，gamma="scale"为sklearn默认自适应gamma
svm_rbf_3d = SVC(kernel="rbf", C=1.0)
# 训练RBF核模型
svm_rbf_3d.fit(X3d_sc, y)
# 预测
pred_3d_rbf = svm_rbf_3d.predict(X3d_sc)
# 打印准确率
print(f"三维RBF-SVM准确率: {accuracy_score(y, pred_3d_rbf):.3f}")
# 打印RBF模型信息
print_svm_info(svm_rbf_3d, X3d_sc, "三维RBF高斯核SVM")
# 绘制3D非线性分隔曲面
plot_3d_decision_surface(svm_rbf_3d, X3d_sc, y,
    "三维数据【沿Z轴可分】| RBF高斯核SVM\n曲面为高维超平面投影")