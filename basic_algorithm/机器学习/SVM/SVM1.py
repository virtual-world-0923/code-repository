import numpy as np  # 导入numpy库用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from sklearn.svm import SVC  # 导入SVM分类器
from sklearn.datasets import make_classification  # 导入数据集生成工具
from sklearn.preprocessing import StandardScaler  # 导入标准化工具
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 设置随机种子，确保结果可重复
np.random.seed(42)


# ==================== 第一部分：生成二维线性可分数据集 ====================
def generate_2d_data():
    """
    生成二维线性可分数据集，包含噪声点
    返回：特征数据X，标签y
    """
    # 使用make_classification生成线性可分数据集
    # n_samples=200：样本数量
    # n_features=2：特征维度为2
    # n_redundant=0：无冗余特征
    # n_informative=2：信息特征为2
    # n_clusters_per_class=1：每个类别一个聚类
    # class_sep=1.5：类别分离程度，值越大类别越容易分开
    # random_state=42：固定随机种子
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )

    # 添加噪声点：随机选择20个样本，翻转其标签
    # np.random.choice(200, 20)：从0-199中随机选择20个索引
    noise_idx = np.random.choice(200, 20, replace=False)
    # 翻转选中的样本标签：1变为0，0变为1
    y[noise_idx] = 1 - y[noise_idx]

    return X, y


# ==================== 第二部分：训练SVM模型 ====================
def train_svm(X, y, kernel='linear', C=1.0):
    """
    训练SVM模型
    参数：
        X：特征数据
        y：标签
        kernel：核函数类型
        C：惩罚参数
    返回：训练好的SVM模型
    """
    # 创建SVC分类器实例
    # kernel='linear'：使用线性核
    # C=1.0：惩罚参数，控制软间隔的容忍度
    # random_state=42：固定随机种子
    svm = SVC(kernel=kernel, C=C, random_state=42)
    # 训练模型
    svm.fit(X, y)
    return svm


# ==================== 第三部分：可视化函数 ====================
def plot_decision_boundary(ax, X, y, svm, title, show_misclassified=False):
    """
    绘制决策边界和分类结果
    参数：
        ax：matplotlib坐标轴对象
        X：特征数据
        y：真实标签
        svm：训练好的SVM模型
        title：图表标题
        show_misclassified：是否显示被错分的样本
    """
    # 创建网格点，用于绘制决策边界
    # 获取X轴和Y轴的范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 生成网格点坐标矩阵
    # np.meshgrid：生成网格点坐标
    # np.arange：生成等差数列
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测网格点的类别
    # np.c_：按列连接数组
    # ravel()：将多维数组展平为一维
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    # 将预测结果重塑为网格形状
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和分类区域
    # contourf：绘制填充等高线图，显示分类区域
    # alpha=0.3：透明度
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # 绘制原始数据点
    # y==0：选择类别0的样本
    # c='blue'：蓝色
    # marker='o'：圆形
    # edgecolors='k'：黑色边框
    # label='Class 0'：图例标签
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o',
               edgecolors='k', label='Class 0', alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s',
               edgecolors='k', label='Class 1', alpha=0.7)

    # 绘制支持向量
    # svm.support_vectors_：获取支持向量
    # facecolors='none'：空心圆
    # edgecolors='green'：绿色边框
    # linewidths=2：线条宽度
    # s=100：大小
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               facecolors='none', edgecolors='green', linewidths=2,
               s=100, label='Support Vectors')

    # 如果show_misclassified为True，绘制被错分的样本
    if show_misclassified:
        # 预测所有样本的类别
        y_pred = svm.predict(X)
        # 找出预测错误的样本
        misclassified = y_pred != y
        # 绘制错分样本
        # marker='x'：叉号
        # s=150：大小
        # color='magenta'：品红色
        ax.scatter(X[misclassified, 0], X[misclassified, 1],
                   marker='x', s=150, color='magenta',
                   label='Misclassified', linewidths=3)

        # 检查错分样本中哪些是支持向量
        # 获取支持向量在原始数据中的索引
        support_indices = svm.support_
        # 获取错分样本的索引
        mis_indices = np.where(misclassified)[0]
        # 找出既是错分样本又是支持向量的索引
        overlap_indices = set(mis_indices) & set(support_indices)

        # 用更大的红圈标记既是错分样本又是支持向量的点
        for idx in overlap_indices:
            ax.scatter(X[idx, 0], X[idx, 1], facecolors='none',
                       edgecolors='red', linewidths=3, s=250)

        # 打印错分样本信息
        print(f"被错分的样本索引: {mis_indices.tolist()}")
        print(f"被错分且是支持向量的样本索引: {list(overlap_indices)}")

    # 设置图表标题
    ax.set_title(title)
    # 设置图例
    ax.legend()
    # 设置坐标轴标签
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # 打印支持向量信息
    print(f"\n{title} 的支持向量信息:")
    print(f"支持向量总数: {len(svm.support_vectors_)}")

    # 打印每个类别的支持向量数量 - 修正后的代码
    print("每个类别的支持向量数量:")
    for i in [0, 1]:
        # 获取支持向量对应的标签
        sv_labels = svm.predict(svm.support_vectors_)
        # 统计类别i的支持向量数量
        count_i = np.sum(sv_labels == i)
        print(f"  类别 {i}: {count_i} 个")

    # 计算样本到超平面的距离
    # decision_function：计算样本到决策超平面的距离
    distances = svm.decision_function(X)

    # 打印每个类别最近的前5个样本的距离
    print("\n每个类别最近的前5个样本距离:")
    for i in [0, 1]:
        # 获取类别i的样本距离
        class_distances = distances[y == i]
        if len(class_distances) > 0:
            # 排序并取前5个最小的距离（最近的样本）
            top5_distances = np.sort(np.abs(class_distances))[:5]
            print(f"  类别 {i}: {top5_distances}")

    # 打印支持向量在原始数据集中的索引
    print(f"\n支持向量在原始数据集中的索引: {svm.support_.tolist()}")


# ==================== 第四部分：生成三维数据集 ====================
def generate_3d_data():
    """
    生成三维线性不可分但二维线性可分的非线性数据集
    返回：特征数据X，标签y
    """
    # 生成200个样本，2个特征，2个类别
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42
    )

    # 添加第三个维度：计算样本到原点的距离的平方，使数据在三维空间中线性可分
    # 计算每个样本到原点的欧几里得距离的平方
    z = np.sum(X ** 2, axis=1)
    # 将z作为第三维特征，形成三维数据
    X_3d = np.column_stack([X, z])

    return X_3d, y


# ==================== 第五部分：三维可视化函数 ====================
def plot_3d_decision(ax, X, y, svm):
    """
    绘制三维决策平面
    参数：
        ax：3D坐标轴对象
        X：三维特征数据
        y：标签
        svm：训练好的SVM模型（使用高斯核）
    """
    # 绘制数据点
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
               c='blue', marker='o', label='Class 0', alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
               c='red', marker='s', label='Class 1', alpha=0.6)

    # 绘制支持向量
    ax.scatter(svm.support_vectors_[:, 0],
               svm.support_vectors_[:, 1],
               svm.support_vectors_[:, 2],
               facecolors='none', edgecolors='green',
               linewidths=2, s=100, label='Support Vectors')

    # 创建网格用于绘制决策平面
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # 生成网格点
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30)
    )

    # 对于每个网格点，我们需要找到z值使得决策函数为0
    # 由于我们在三维空间中使用RBF核，决策边界是非线性的
    # 这里我们只绘制一个近似的决策平面：使用SVM的决策函数在网格上的值
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 为了绘制决策平面，我们需要在z维度上搜索决策边界
    z_vals = np.linspace(X[:, 2].min() - 0.5, X[:, 2].max() + 0.5, 20)
    z_grid = np.zeros_like(xx)

    # 对每个(x,y)点，找到决策函数值为0的z值
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_val = xx[i, j]
            y_val = yy[i, j]
            # 在z维度上搜索决策边界
            for z_val in z_vals:
                test_point = np.array([[x_val, y_val, z_val]])
                # 使用SVM的决策函数
                decision_val = svm.decision_function(test_point)[0]
                if abs(decision_val) < 0.1:  # 找到决策函数接近0的点
                    z_grid[i, j] = z_val
                    break
            else:
                z_grid[i, j] = np.nan  # 如果没有找到，设为NaN

    # 绘制决策平面
    # plot_surface：绘制3D曲面
    # alpha=0.3：透明度
    # cmap='coolwarm'：颜色映射
    surf = ax.plot_surface(xx, yy, z_grid, alpha=0.3, cmap='coolwarm')

    # 设置坐标轴标签
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3 (z)')
    ax.set_title('3D SVM with Gaussian Kernel')
    ax.legend()


# ==================== 主程序 ====================
def main():
    """
    主函数：执行整个SVM分类演示流程
    """
    # 1. 生成二维线性可分数据集
    print("=" * 50)
    print("生成二维线性可分数据集...")
    X_2d, y_2d = generate_2d_data()
    print(f"数据集大小: {X_2d.shape[0]} 个样本, {X_2d.shape[1]} 个特征")
    print(f"类别分布: 类别0有 {np.sum(y_2d == 0)} 个样本, 类别1有 {np.sum(y_2d == 1)} 个样本")

    # 2. 训练硬间隔SVM模型
    print("\n" + "=" * 50)
    print("训练硬间隔SVM模型 (C=1000, 近似硬间隔)...")
    # C值很大时，近似于硬间隔分类器
    svm_hard = train_svm(X_2d, y_2d, kernel='linear', C=1000)

    # 3. 训练软间隔SVM模型
    print("\n" + "=" * 50)
    print("训练软间隔SVM模型 (C=0.5)...")
    # C值较小时，允许更多的误分类，实现软间隔
    svm_soft = train_svm(X_2d, y_2d, kernel='linear', C=0.5)

    # 4. 创建图形窗口
    fig = plt.figure(figsize=(16, 12))

    # 5. 绘制硬间隔和软间隔对比
    ax1 = fig.add_subplot(2, 2, 1)
    plot_decision_boundary(ax1, X_2d, y_2d, svm_hard,
                           'Hard Margin SVM (C=1000)', show_misclassified=False)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_decision_boundary(ax2, X_2d, y_2d, svm_soft,
                           'Soft Margin SVM (C=0.5)', show_misclassified=True)

    # 6. 生成三维非线性数据集
    print("\n" + "=" * 50)
    print("生成三维非线性数据集...")
    X_3d, y_3d = generate_3d_data()
    print(f"三维数据集大小: {X_3d.shape[0]} 个样本, {X_3d.shape[1]} 个特征")

    # 7. 标准化三维数据
    scaler = StandardScaler()
    X_3d_scaled = scaler.fit_transform(X_3d)

    # 8. 训练高斯核SVM模型
    print("\n" + "=" * 50)
    print("训练高斯核SVM模型 (RBF核)...")
    # kernel='rbf'：使用高斯径向基核函数
    # gamma='scale'：核函数系数
    # C=1.0：惩罚参数
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_3d_scaled, y_3d)
    print(f"支持向量数量: {len(svm_rbf.support_vectors_)}")
    print(f"支持向量索引: {svm_rbf.support_.tolist()}")
    print(f"训练准确率: {svm_rbf.score(X_3d_scaled, y_3d):.4f}")

    # 9. 绘制三维分类结果
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_3d_decision(ax3, X_3d_scaled, y_3d, svm_rbf)

    # 10. 调整布局并显示
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("SVM分类演示完成！")


# 如果作为主程序运行，执行main函数
if __name__ == "__main__":
    main()