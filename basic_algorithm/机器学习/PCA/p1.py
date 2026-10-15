# ==================== 导入所需库 ====================

# 导入numpy库，用于数值计算、矩阵运算（版本2.5.1）
import numpy as np

# 导入pandas库，用于读取和处理结构化数据文件（版本3.0.3）
import pandas as pd

# 从sklearn.decomposition模块导入PCA类，用于执行主成分分析（sklearn版本1.9.0）
from sklearn.decomposition import PCA

# 从sklearn.preprocessing模块导入StandardScaler类，用于数据标准化（均值为0，方差为1）
from sklearn.preprocessing import StandardScaler


# ==================== 第一步：加载数据集 ====================

# 定义鸢尾花数据集的文件路径（UCI标准格式：无表头，逗号分隔，最后一列为类别标签）
file_path = r"D:\Repositories\basic_algorithm\database\iris.data"

# 定义列名列表，前4列为数值特征，第5列为类别标签
# 参数说明：依次为花萼长度(cm)、花萼宽度(cm)、花瓣长度(cm)、花瓣宽度(cm)、品种名称
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 使用pandas的read_csv函数读取CSV数据文件
# 参数说明：
#   file_path: 数据文件的完整路径字符串
#   header=None: 指定文件没有表头行（第一行就是数据），需要手动指定列名
#   names=column_names: 传入自定义列名列表，为DataFrame的每列赋予名称
iris_data = pd.read_csv(file_path, header=None, names=column_names)

# 打印数据集基本信息：形状（样本数, 列数）
# 参数说明：shape是DataFrame属性，返回元组(150, 5)，即150个样本、5列
print("=" * 70)
print("【数据集基本信息】")
print(f"  数据集形状: {iris_data.shape}")
print(f"  列名: {iris_data.columns.tolist()}")


# ==================== 第二步：分离特征矩阵和标签 ====================

# 使用iloc位置索引选取前4列（索引0~3）作为特征矩阵，并转为numpy数组
# 参数说明：
#   iloc[:, :4]: 第一个冒号=所有行，:4=第0到第3列（共4个数值特征）
#   .values: 将pandas DataFrame转换为numpy ndarray，形状为(150, 4)
X = iris_data.iloc[:, :4].values

# 使用iloc选取第5列（索引4）作为标签向量，并转为numpy数组
# 参数说明：
#   iloc[:, 4]: 所有行的第4列（class列）
#   .values: 转为numpy数组，形状为(150,)
y = iris_data.iloc[:, 4].values

# 定义特征名称列表，用于后续输出时显示可读的特征名
# 参数说明：与column_names前4个元素对应，分别是4个原始特征的英文名
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


# ==================== 第三步：输出原始第一条数据 ====================

print("\n" + "=" * 70)
print("【一、原始第一条数据】")
print("=" * 70)

# 获取特征矩阵的第一行（索引0），即第一个样本的4个特征值
# 参数说明：X[0]返回形状为(4,)的一维数组，包含该样本的4个原始特征值
first_sample_original = X[0]

# 打印第一条原始数据的各特征值
# 参数说明：
#   feature_names[i]: 第i个特征的名称字符串
#   first_sample_original[i]: 第i个特征的数值
#   .tolist(): 将numpy数组转为Python原生list，避免numpy 2.5.1显示np.float64(...)前缀
print(f"  原始第一条数据: {first_sample_original.tolist()}")

# 逐特征打印，更清晰地展示每个特征名和对应值
# 参数说明：zip将特征名列表和特征值数组配对，name为特征名，val为对应数值
for name, val in zip(feature_names, first_sample_original):
    # 打印每个特征的名称和数值，保留4位小数
    print(f"    {name}: {val:.4f}")


# ==================== 第四步：输出原始数据的协方差矩阵 ====================

print("\n" + "=" * 70)
print("【二、原始数据的协方差矩阵】")
print("=" * 70)

# 使用numpy的cov函数计算原始特征矩阵的协方差矩阵
# 参数说明：
#   X: 传入形状为(150, 4)的特征矩阵
#   rowvar=False: 指定每列代表一个变量（特征），每行代表一个观测（样本）
#                 如果rowvar=True（默认），则每行代表一个变量
#   返回值: 形状为(4, 4)的对称协方差矩阵，对角线为各特征方差，非对角线为特征间协方差
cov_original = np.cov(X, rowvar=False)

# 打印协方差矩阵，使用round保留4位小数
# 参数说明：np.round(cov_original, 4)将矩阵每个元素四舍五入到4位小数
print(f"  协方差矩阵形状: {cov_original.shape}")
print(f"  协方差矩阵（4×4）:")
print(f"  {'':>14s} {'sepal_length':>12s} {'sepal_width':>12s} {'petal_length':>12s} {'petal_width':>12s}")

# 使用for循环逐行打印协方差矩阵，附带特征名称作为行标签
# 参数说明：
#   enumerate(cov_original): 对协方差矩阵按行枚举，i为行索引，row为该行数组
#   feature_names[i]: 当前行的特征名称
#   row.tolist(): 转为Python list避免numpy 2.5.1的np.float64显示问题
for i, row in enumerate(cov_original):
    # 格式化输出：行标签 + 4个协方差值（每个占12字符宽度，保留4位小数）
    print(f"  {feature_names[i]:>14s} {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f} {row[3]:>12.4f}")

# 打印对角线元素（即各特征的方差）
# 参数说明：np.diag(cov_original)提取协方差矩阵的对角线元素，返回长度为4的数组
print(f"\n  各特征方差（对角线）: {np.diag(cov_original).round(4).tolist()}")


# ==================== 第五步：数据标准化 ====================

print("\n" + "=" * 70)
print("【三、数据标准化】")
print("=" * 70)

# 创建StandardScaler标准化器实例
# 参数说明：StandardScaler()无需参数，默认将每列转换为均值=0、标准差=1
# 作用：PCA对数据尺度敏感，标准化消除不同特征量纲差异（如cm vs mm）
scaler = StandardScaler()

# 对特征矩阵执行标准化（拟合并转换一步完成）
# 参数说明：
#   X: 传入原始特征矩阵，形状(150, 4)
#   返回值: 标准化后的矩阵，形状不变(150, 4)，每列均值≈0，标准差≈1
# 内部计算：X_scaled = (X - mean) / std
X_scaled = scaler.fit_transform(X)

# 打印标准化后第一条数据，验证标准化效果
# 参数说明：X_scaled[0].tolist()取第一行并转为Python list
print(f"  标准化后第一条数据: {X_scaled[0].round(4).tolist()}")
print(f"  标准化后各特征均值: {X_scaled.mean(axis=0).round(6).tolist()}")
print(f"  标准化后各特征标准差: {X_scaled.std(axis=0).round(4).tolist()}")


# ==================== 第六步：使用PCA（SVD方式）拟合全部主成分 ====================

print("\n" + "=" * 70)
print("【四、PCA拟合（SVD分解方式）】")
print("=" * 70)

# 创建PCA实例，保留全部4个主成分，使用完整SVD分解
# 参数说明：
#   n_components=4: 保留4个主成分（即全部保留，用于分析所有主成分信息）
#   svd_solver='full': 使用完整奇异值分解（scipy.linalg.svd），精确计算所有主成分
#                      适合小数据集（如鸢尾花150×4），计算速度快且精确
#   copy=True（默认）: 拟合时复制输入数据，不修改原始数组
#   whiten=False（默认）: 不进行白化处理，保留主成分的原始方差
#   random_state=None（默认）: 不设置随机种子（full模式不需要）
pca_svd = PCA(n_components=4, svd_solver='full')

# 对标准化后的数据执行PCA拟合（仅计算主成分方向和方差，不转换数据）
# 参数说明：
#   X_scaled: 标准化后的特征矩阵，形状(150, 4)
#   内部过程：
#     1. 计算X_scaled的均值（已接近0）
#     2. 中心化：X_centered = X_scaled - mean_
#     3. SVD分解：X_centered = U @ diag(σ) @ V^T
#     4. components_ = V（主成分方向）
#     5. explained_variance_ = σ² / (n-1)
pca_svd.fit(X_scaled)

# 打印使用的分解方式
print(f"  分解方式: SVD（奇异值分解）")
print(f"  svd_solver: {pca_svd.svd_solver}")


# ==================== 第七步：输出PCA变换后的第一条数据 ====================

print("\n" + "=" * 70)
print("【五、PCA变换后的第一条数据】")
print("=" * 70)

# 使用transform方法将标准化数据投影到主成分空间
# 参数说明：
#   X_scaled: 传入标准化后的特征矩阵，形状(150, 4)
#   返回值: 投影后的数据矩阵，形状(150, 4)，每行是样本在主成分空间的坐标
#   内部计算：X_pca = (X_scaled - mean_) @ components_.T
X_pca_full = pca_svd.transform(X_scaled)

# 获取变换后的第一条数据（索引0）
# 参数说明：X_pca_full[0]返回形状为(4,)的数组，包含该样本在4个主成分上的投影坐标
first_sample_pca = X_pca_full[0]

# 打印变换后的第一条数据
# 参数说明：.round(4).tolist()四舍五入到4位小数并转为Python list
print(f"  变换后第一条数据（4维主成分坐标）: {first_sample_pca.round(4).tolist()}")

# 逐主成分打印变换后的值
# 参数说明：enumerate对数组枚举，i为主成分索引（0~3），val为投影坐标值
for i, val in enumerate(first_sample_pca):
    # 打印第i+1个主成分的投影值
    print(f"    PC{i+1}: {val:.4f}")

# 手动验证：用公式 (X - mean_) @ components_.T 计算第一条数据的变换结果
# 参数说明：
#   X_scaled[0:1]: 取第一行，保持二维形状(1, 4)，便于矩阵乘法
#   pca_svd.mean_: 各特征均值数组，形状(4,)
#   pca_svd.components_.T: 主成分方向矩阵的转置，形状(4, 4)
first_manual = (X_scaled[0:1] - pca_svd.mean_) @ pca_svd.components_.T

# 验证手动计算与transform结果是否一致
# 参数说明：np.allclose比较两个数组是否在容差范围内相等
print(f"\n  手动验证 (X-mean)@V^T: {first_manual[0].round(4).tolist()}")
print(f"  与transform结果一致: {np.allclose(first_sample_pca, first_manual[0])}")


# ==================== 第八步：输出变换后数据的协方差矩阵 ====================

print("\n" + "=" * 70)
print("【六、变换后数据的协方差矩阵】")
print("=" * 70)

# 计算PCA变换后数据的协方差矩阵
# 参数说明：
#   X_pca_full: 变换后的数据矩阵，形状(150, 4)
#   rowvar=False: 每列为一个变量（主成分），每行为一个观测
#   返回值: 形状(4, 4)的协方差矩阵
#   重要性质：PCA变换后的协方差矩阵应为对角矩阵（非对角线元素≈0），
#            因为主成分之间是正交的（不相关）
cov_pca = np.cov(X_pca_full, rowvar=False)

# 打印变换后的协方差矩阵
# 参数说明：round(4)保留4位小数，tolist()避免numpy 2.5.1的显示问题
print(f"  变换后协方差矩阵形状: {cov_pca.shape}")
print(f"  变换后协方差矩阵（应为近似对角矩阵）:")
print(f"  {'':>6s} {'PC1':>12s} {'PC2':>12s} {'PC3':>12s} {'PC4':>12s}")

# 逐行打印变换后的协方差矩阵
# 参数说明：enumerate按行枚举，i为行索引，row为该行数组
for i, row in enumerate(cov_pca):
    # 格式化输出：PC标签 + 4个协方差值
    print(f"  PC{i+1} {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f} {row[3]:>12.4f}")

# 打印对角线（即各主成分的方差(特征值) = explained_variance_）
# 参数说明：np.diag提取对角线元素
print(f"\n  对角线（各主成分方差）: {np.diag(cov_pca).round(4).tolist()}")
print(f"  非对角线最大值（应≈0）: {np.max(np.abs(cov_pca - np.diag(np.diag(cov_pca)))):.2e}")


# ==================== 第九步：输出每个主成分的贡献度（从大到小排序） ====================

print("\n" + "=" * 70)
print("【七、每个主成分的贡献度（从大到小排序）】")
print("=" * 70)

# 获取各主成分的方差解释比例（贡献度），已按从大到小排序
# 参数说明：
#   explained_variance_ratio_: PCA拟合后的属性，返回长度为n_components的数组
#   每个元素表示对应主成分解释的方差占总方差的比例
#   sklearn内部已按特征值降序排列，无需额外排序
variance_ratio = pca_svd.explained_variance_ratio_

# 获取各主成分的实际方差值（特征值）
# 参数说明：explained_variance_返回长度为n_components的数组，即协方差矩阵的特征值
variance_values = pca_svd.explained_variance_

# 打印表头
# 参数说明：格式化字符串，定义列宽和对齐方式
print(f"  {'主成分':<10s} {'贡献度(%)':<12s} {'累计贡献度(%)':<14s} {'方差值(特征值)':<14s}")
print(f"  {'-'*50}")

# 使用循环逐个打印每个主成分的贡献度信息
# 参数说明：
#   enumerate(variance_ratio): 枚举贡献度数组，i为索引(0~3)，ratio为贡献度值
#   np.cumsum(variance_ratio): 计算累计贡献度数组
cumulative_ratio = np.cumsum(variance_ratio)

# 遍历每个主成分，打印详细信息
for i, ratio in enumerate(variance_ratio):
    # 打印：主成分名称、贡献度百分比、累计贡献度百分比、方差值
    # 参数说明：
    #   f"PC{i+1}": 主成分编号（从1开始）
    #   ratio*100: 贡献度转为百分比
    #   cumulative_ratio[i]*100: 累计贡献度转为百分比
    #   variance_values[i]: 该主成分的方差值（特征值）
    print(f"  PC{i+1:<8d} {ratio*100:<12.2f} {cumulative_ratio[i]*100:<14.2f} {variance_values[i]:<14.4f}")


# ==================== 第十步：输出奇异值、特征值、特征向量 ====================

print("\n" + "=" * 70)
print("【八、奇异值、特征值、特征向量（从大到小排序）】")
print("=" * 70)

# ---------- 8.1 奇异值 ----------
# 获取PCA分解得到的奇异值数组
# 参数说明：
#   singular_values_: PCA拟合后的属性，返回长度为n_components的数组
#   奇异值是SVD分解 X = U @ diag(σ) @ V^T 中的σ值
#   已按从大到小排序
#   与方差的关系：variance = σ² / (n_samples - 1)
singular_values = pca_svd.singular_values_

# 打印奇异值
# 参数说明：.round(4).tolist()四舍五入并转为Python list
print(f"\n  【奇异值 σ（从大到小）】")
print(f"  σ = {singular_values.round(4).tolist()}")

# 逐个打印奇异值及其含义
for i, sv in enumerate(singular_values):
    # 打印第i+1个奇异值
    print(f"    σ{i+1} = {sv:.4f}")

# ---------- 8.2 特征值 ----------
# 特征值即explained_variance_，是协方差矩阵的特征值
# 参数说明：
#   explained_variance_: 长度为n_components的数组
#   数学关系：λ_i = σ_i² / (n_samples - 1)
#   已按从大到小排序
eigenvalues = pca_svd.explained_variance_

# 打印特征值
print(f"\n  【特征值 λ（从大到小）】")
print(f"  λ = {eigenvalues.round(4).tolist()}")

# 逐个打印特征值
for i, ev in enumerate(eigenvalues):
    # 打印第i+1个特征值
    print(f"    λ{i+1} = {ev:.4f}")

# 验证奇异值与特征值的数学关系：λ = σ² / (n-1)
# 参数说明：
#   singular_values**2: 奇异值的平方
#   X_scaled.shape[0] - 1: 样本数减1 = 150 - 1 = 149
#   这是无偏估计的分母（贝塞尔校正）
eigenvalues_from_sv = singular_values**2 / (X_scaled.shape[0] - 1)
print(f"\n  验证 λ = σ²/(n-1): {eigenvalues_from_sv.round(4).tolist()}")
print(f"  与explained_variance_一致: {np.allclose(eigenvalues, eigenvalues_from_sv)}")

# ---------- 8.3 特征向量（主成分方向/载荷矩阵） ----------
# 获取主成分方向矩阵（即协方差矩阵的特征向量）
# 参数说明：
#   components_: PCA拟合后的属性，形状为(n_components, n_features_in_) = (4, 4)
#   每一行是一个主成分方向向量（特征向量）
#   components_[i] 对应第i+1个主成分的方向
#   已按对应特征值从大到小排序
eigenvectors = pca_svd.components_

# 打印特征向量矩阵
print(f"\n  【特征向量（主成分方向矩阵，每行为一个主成分方向）】")
print(f"  形状: {eigenvectors.shape}（4个主成分 × 4个原始特征）")
print(f"\n  {'':>6s} {'sepal_length':>14s} {'sepal_width':>14s} {'petal_length':>14s} {'petal_width':>14s}")

# 逐行打印特征向量（每个主成分在4个原始特征上的载荷）
# 参数说明：enumerate按行枚举，i为主成分索引，vec为该主成分的方向向量
for i, vec in enumerate(eigenvectors):
    # 打印第i+1个主成分的方向向量，每个分量保留4位小数
    print(f"  PC{i+1}  {vec[0]:>14.4f} {vec[1]:>14.4f} {vec[2]:>14.4f} {vec[3]:>14.4f}")


# ==================== 第十一步：手动使用协方差矩阵特征分解验证 ====================

print("\n" + "=" * 70)
print("【九、手动协方差矩阵特征分解验证】")
print("=" * 70)

# 计算标准化数据的协方差矩阵
# 参数说明：
#   X_scaled: 标准化后的数据，形状(150, 4)
#   rowvar=False: 每列为一个变量
#   返回值: 形状(4, 4)的协方差矩阵
cov_matrix = np.cov(X_scaled, rowvar=False)

# 使用numpy的eigh函数对协方差矩阵进行特征分解
# 参数说明：
#   cov_matrix: 传入4×4的实对称协方差矩阵
#   返回值:
#     eigen_vals: 特征值数组，形状(4,)，按升序排列（eigh默认升序）
#     eigen_vecs: 特征向量矩阵，形状(4, 4)，每列为一个特征向量
#   注意：eigh专门用于实对称矩阵/Hermitian矩阵，比eig更快更稳定
#   注意：eigh返回的特征值是升序的，需要手动反转为降序
eigen_vals_manual, eigen_vecs_manual = np.linalg.eigh(cov_matrix)

# 将特征值和特征向量从升序反转为降序（从大到小）
# 参数说明：
#   [::-1]: 将数组/矩阵的行或列顺序反转
#   eigen_vals_manual[::-1]: 特征值从升序变为降序
#   eigen_vecs_manual[:, ::-1]: 特征向量矩阵的列顺序反转，与特征值对应
idx_desc = np.argsort(eigen_vals_manual)[::-1]
eigen_vals_desc = eigen_vals_manual[idx_desc]
eigen_vecs_desc = eigen_vecs_manual[:, idx_desc]

# 打印手动分解的特征值（降序）
print(f"  手动特征分解 - 特征值（降序）: {eigen_vals_desc.round(4).tolist()}")
print(f"  sklearn特征值（降序）:          {eigenvalues.round(4).tolist()}")
print(f"  特征值一致: {np.allclose(eigen_vals_desc, eigenvalues)}")

# 打印手动分解的特征向量（取绝对值比较，因为特征向量方向可正可负）
# 参数说明：
#   np.abs(): 取绝对值，消除方向符号差异
#   eigenvectors: sklearn的components_，每行为一个特征向量
#   eigen_vecs_desc.T: 手动分解的特征向量矩阵转置（每行变为一个特征向量）
print(f"\n  特征向量方向一致（取绝对值比较）: {np.allclose(np.abs(eigenvectors), np.abs(eigen_vecs_desc.T))}")

# 手动计算奇异值：σ = sqrt(λ * (n-1))
# 参数说明：
#   eigen_vals_desc: 降序排列的特征值
#   X_scaled.shape[0] - 1: n-1 = 149
#   np.sqrt(): 开平方根
singular_manual = np.sqrt(eigen_vals_desc * (X_scaled.shape[0] - 1))
print(f"\n  手动计算奇异值: {singular_manual.round(4).tolist()}")
print(f"  sklearn奇异值:  {singular_values.round(4).tolist()}")
print(f"  奇异值一致: {np.allclose(singular_manual, singular_values)}")


# ==================== 第十二步：PCA降维到2维 ====================

print("\n" + "=" * 70)
print("【十、PCA降维到2维】")
print("=" * 70)

# ---------- 方式一：使用SVD分解降维 ----------
# 创建PCA实例，指定保留2个主成分，使用完整SVD
# 参数说明：
#   n_components=2: 只保留贡献度最大的前2个主成分
#   svd_solver='full': 使用完整SVD分解（scipy.linalg.svd）
#                      对150×4的小数据集，full模式精确且快速
#   其他参数使用默认值：copy=True, whiten=False, tol=0.0,
#                      iterated_power='auto', random_state=None
pca_2d_svd = PCA(n_components=2, svd_solver='full')

# 对标准化数据执行拟合并转换（一步完成降维）
# 参数说明：
#   X_scaled: 标准化后的特征矩阵，形状(150, 4)
#   返回值: 降维后的数据矩阵，形状(150, 2)
#   内部过程：fit计算主成分 → transform投影到前2个主成分
X_pca_2d_svd = pca_2d_svd.fit_transform(X_scaled)

# 打印SVD方式降维结果
print(f"\n  【方式一：SVD分解】")
print(f"  降维后数据形状: {X_pca_2d_svd.shape}")
print(f"  第一条数据（2维坐标）: {X_pca_2d_svd[0].round(4).tolist()}")

# ---------- 方式二：使用协方差矩阵特征分解降维 ----------
# 创建PCA实例，指定保留2个主成分，使用协方差矩阵特征分解
# 参数说明：
#   n_components=2: 只保留前2个主成分
#   svd_solver='covariance_eigh': 使用协方差矩阵 + numpy.linalg.eigh 进行特征分解
#                                 先计算 C = X^T @ X / (n-1)，再对C做特征分解
#                                 适合特征数较少的场景（如本例4个特征）
#                                 sklearn 1.2+ 支持此选项，1.9.0完全可用
pca_2d_eigh = PCA(n_components=2, svd_solver='covariance_eigh')

# 对标准化数据执行拟合并转换
# 参数说明：同方式一，X_scaled形状(150, 4)，返回形状(150, 2)
X_pca_2d_eigh = pca_2d_eigh.fit_transform(X_scaled)

# 打印协方差矩阵特征分解方式降维结果
print(f"\n  【方式二：协方差矩阵特征分解（covariance_eigh）】")
print(f"  降维后数据形状: {X_pca_2d_eigh.shape}")
print(f"  第一条数据（2维坐标）: {X_pca_2d_eigh[0].round(4).tolist()}")

# 验证两种方式的结果是否一致（取绝对值，因为主成分方向可能相反）
# 参数说明：
#   np.abs(): 取绝对值消除方向符号差异
#   np.allclose(): 判断两个数组在容差范围内是否相等
print(f"\n  两种方式结果一致（取绝对值）: {np.allclose(np.abs(X_pca_2d_svd), np.abs(X_pca_2d_eigh))}")


# ==================== 第十三步：输出2维表示原始信息的百分比 ====================

print("\n" + "=" * 70)
print("【十一、2维主成分表示原始信息的百分比】")
print("=" * 70)

# 获取降维后PCA模型的方差解释比例（2个主成分的贡献度）
# 参数说明：
#   pca_2d_svd.explained_variance_ratio_: 返回长度为2的数组
#   [0]为PC1的贡献度，[1]为PC2的贡献度
ratio_2d = pca_2d_svd.explained_variance_ratio_

# 计算2个主成分的累计贡献度（即保留的信息百分比）
# 参数说明：np.sum()对数组所有元素求和
total_info_ratio = np.sum(ratio_2d)

# 打印每个主成分的贡献度
print(f"\n  PC1 贡献度: {ratio_2d[0]*100:.2f}%")
print(f"  PC2 贡献度: {ratio_2d[1]*100:.2f}%")

# 打印累计信息保留百分比
print(f"\n  ┌─────────────────────────────────────────────┐")
print(f"  │  2维主成分累计贡献度: {total_info_ratio*100:.2f}%              │")
print(f"  │  即：降维到2维后保留了原始4维数据 {total_info_ratio*100:.2f}% 的信息 │")
print(f"  │  信息损失: {(1-total_info_ratio)*100:.2f}%                          │")
print(f"  └─────────────────────────────────────────────┘")

# 打印降维后数据的协方差矩阵（验证正交性）
# 参数说明：np.cov计算2维数据的协方差矩阵，形状(2, 2)
cov_2d = np.cov(X_pca_2d_svd, rowvar=False)
print(f"\n  降维后协方差矩阵（2×2，应为对角矩阵）:")
print(f"    [{cov_2d[0][0]:>8.4f}  {cov_2d[0][1]:>8.4f}]")
print(f"    [{cov_2d[1][0]:>8.4f}  {cov_2d[1][1]:>8.4f}]")
print(f"  非对角线元素（应≈0）: {cov_2d[0][1]:.2e}")

# 打印降维后第一条数据
print(f"\n  降维后第一条数据: {X_pca_2d_svd[0].round(4).tolist()}")
print(f"  对比原始第一条数据: {X[0].tolist()}")


# ==================== 第十四步：完整总结 ====================

print("\n" + "=" * 70)
print("【完整总结】")
print("=" * 70)
print(f"  原始数据维度:     {X.shape[1]} 维（{X.shape[0]}个样本）")
print(f"  降维后维度:       {X_pca_2d_svd.shape[1]} 维")
print(f"  分解方式:         SVD / 协方差矩阵特征分解（两者等价）")
print(f"  信息保留率:       {total_info_ratio*100:.2f}%")
print(f"  PC1贡献:          {ratio_2d[0]*100:.2f}%（主要反映花瓣/花萼长度信息）")
print(f"  PC2贡献:          {ratio_2d[1]*100:.2f}%（主要反映花萼宽度信息）")
print(f"  降维完成！")