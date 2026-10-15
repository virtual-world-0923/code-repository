# 导入numpy库，用于执行高效的数值计算、矩阵运算及线性代数操作
import numpy as np
# 导入pandas库，用于读取结构化数据文件（如CSV）并进行数据预处理
import pandas as pd

# ==================== 1. 加载数据集 ====================
# 使用pandas的read_csv函数读取指定路径下的鸢尾花数据集文件
# 参数说明：
#   filepath_or_buffer: 字符串，指定文件的绝对路径 "D:\Repositories\basic_algorithm\database\iris.data"
#   header: None，表示文件第一行不是列名，而是数据的一部分
#   names: 列表，手动定义5个列名，分别对应4个特征和1个类别标签
df = pd.read_csv(
    r"D:\Repositories\basic_algorithm\database\iris.data",
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
)

# ==================== 2. 提取特征矩阵 ====================
# 从DataFrame中提取前4列数值型特征数据，排除最后一列类别标签
# 参数说明：
#   df.iloc[:, :4]: 使用iloc位置索引，选取所有行（:）和前4列（:4）
#   .values: 将Pandas DataFrame转换为NumPy ndarray数组，以便进行矩阵运算
X = df.iloc[:, :4].values

# ==================== 3. 数据中心化（Zero-mean Normalization） ====================
# 计算每个特征的均值，并从原始数据中减去该均值，使数据分布中心移至原点
# 参数说明：
#   X.mean(axis=0): 沿轴0（列方向）计算均值，返回形状为(4,)的一维数组，包含4个特征的均值
#   X - ...: 利用NumPy广播机制，将每行数据减去对应的特征均值
X_centered = X - X.mean(axis=0)


# ==================== 4. 定义手写PCA函数 ====================
def manual_pca(data, n_components):
    """
    基于协方差矩阵特征值分解的手写PCA算法。

    参数说明：
        data: NumPy数组，形状为(n_samples, n_features)，必须是已经中心化的数据
        n_components: 整数，指定降维后的目标维度数

    返回：
        projected_data: 降维后的数据，形状为(n_samples, n_components)
        eigenvalues: 排序后的特征值数组，形状为(n_features,)
        eigenvectors: 排序后的特征向量矩阵，形状为(n_features, n_features)
    """
    # 步骤1: 计算协方差矩阵
    # 参数说明：
    #   data.T: 转置数据矩阵，形状变为(n_features, n_samples)，因为np.cov默认每行是一个变量
    #   np.cov(...): 计算协方差矩阵，返回形状为(n_features, n_features)的对称矩阵
    cov_matrix = np.cov(data.T)

    # 步骤2: 对协方差矩阵进行特征值分解
    # 参数说明：
    #   np.linalg.eigh(cov_matrix): 专门用于实对称矩阵的特征值分解，比通用eig更稳定且速度更快
    #   返回值eigenvalues: 一维数组，包含所有特征值（默认按升序排列）
    #   返回值eigenvectors: 二维数组，每一列是对应特征值的单位特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 步骤3: 将特征值和特征向量按特征值从大到小排序
    # 参数说明：
    #   np.argsort(eigenvalues): 返回特征值从小到大排序的索引数组
    #   [::-1]: 切片操作，反转索引数组，使其变为从大到小排序
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # 根据排序索引重新排列特征值
    # 参数说明：
    #   eigenvalues[sorted_indices]: 使用高级索引获取降序排列的特征值
    eigenvalues_sorted = eigenvalues[sorted_indices]

    # 根据排序索引重新排列特征向量矩阵的列
    # 参数说明：
    #   eigenvectors[:, sorted_indices]: 选取所有行，但列按照降序特征值的索引排列
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # 步骤4: 构建投影矩阵并执行降维
    # 参数说明：
    #   eigenvectors_sorted[:, :n_components]: 选取前n_components个最大特征值对应的特征向量，形状为(n_features, n_components)
    projection_matrix = eigenvectors_sorted[:, :n_components]

    # 将中心化后的数据投影到主成分空间
    # 参数说明：
    #   np.dot(data, projection_matrix): 矩阵乘法，(n_samples, n_features) x (n_features, n_components) -> (n_samples, n_components)
    projected_data = np.dot(data, projection_matrix)

    # 返回降维数据、完整排序后的特征值和特征向量矩阵
    return projected_data, eigenvalues_sorted, eigenvectors_sorted


# ==================== 5. 执行PCA降维 ====================
# 调用自定义PCA函数，将4维数据降维至2维
# 参数说明：
#   X_centered: 已中心化的原始特征数据
#   n_components=2: 指定保留前2个主成分
X_pca, eigenvalues, eigenvectors = manual_pca(X_centered, n_components=2)

# 验证A · v = λ · v
A = cov_matrix = np.cov(X_centered.T)
v = eigenvectors[:, 2]       # 取第0列作为特征向量
lam = eigenvalues[2]         # 取第0个特征值

# 以下两者应相等（允许浮点误差）
print('-'*100)
print(A @ v)                 # 矩阵乘以特征向量
print('-'*100)
print(lam * v)               # 特征值乘以特征向量
print('-'*100)

# ==================== 6. 输出分解后的三个矩阵/数组 ====================
# 打印特征值数组（已按降序排列）
print("特征值 (Eigenvalues):")
print(eigenvalues)

# 打印特征向量矩阵（每列为一个特征向量，已按对应特征值降序排列）
print("\n特征向量矩阵 (Eigenvectors):")
print(eigenvectors)

# 计算并打印协方差矩阵（为了展示“三个矩阵”中的协方差矩阵，这里重新计算或引用内部逻辑结果）
# 注意：通常PCA分解直接产出的是特征值和特征向量，协方差矩阵是中间产物
cov_matrix_result = np.cov(X_centered.T)
print("\n协方差矩阵 (Covariance Matrix):")
print(cov_matrix_result)

# ==================== 7. 输出原始第一条数据值 ====================
# 从原始未处理的数据矩阵X中获取第一行（索引为0）
# 参数说明：
#   X[0]: 访问NumPy数组的第一行，返回包含4个特征值的数组
print("\n原始第一条数据值:")
print(X[0])

# ==================== 8. 输出降维后的第一条数据值 ====================
# 从降维后的结果矩阵X_pca中获取第一行（索引为0）
# 参数说明：
#   X_pca[0]: 访问降维后数组的第一行，返回包含2个主成分得分的数组
print("\n降维后的第一条数据值:")
print(X_pca[0])