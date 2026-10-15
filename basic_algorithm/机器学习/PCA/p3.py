# 导入必要的库
import numpy as np  # 用于数值计算和数组操作，版本 2.5.1
import pandas as pd  # 用于数据处理和读取 CSV 文件，版本 3.0.3
from sklearn.decomposition import PCA  # 从 sklearn 中导入 PCA 类，版本 1.9.0

# 设置数据文件路径，使用原始字符串避免反斜杠转义问题
file_path = r"D:\Repositories\basic_algorithm\database\iris.data"

# 定义鸢尾花数据集的列名（该数据集原始没有表头）
# 前四列为特征，最后一列为类别标签
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

# 使用 pandas 读取 CSV 文件
# 参数说明：
#   file_path: 文件路径
#   header=None: 表示原始数据没有列名行，需要自行指定
#   names=column_names: 为数据框添加指定的列名
df = pd.read_csv(file_path, header=None, names=column_names)

# 提取特征矩阵 X，取所有行的前四列（即四个数值型特征）
# 参数说明：iloc[:, :-1] 表示所有行，列从开始到倒数第二列（不包含最后一列）
X = df.iloc[:, :-1]

# 提取类别标签 y，取所有行的最后一列（此处仅读取，后续分析未直接使用）
# 参数说明：iloc[:, -1] 表示所有行，只取最后一列
y = df.iloc[:, -1]

# =============================================================================
# 1. 输出原始第一条数据值
# 使用 iloc[0] 获取第一行数据（包含所有特征列）
# 参数说明：iloc[0] 表示按位置索引，取出索引为0的行（即第一行）
print("原始第一条数据值：")
print(X.iloc[0])

# =============================================================================
# 2. 计算并输出原始数据协方差矩阵
# 使用 DataFrame 的 cov() 方法计算特征之间的协方差矩阵
# 协方差矩阵反映了原始四个特征两两之间的线性相关程度
cov_original = X.cov()
print("\n原始数据协方差矩阵：")
print(cov_original)

# =============================================================================
# 3. 使用完整 PCA（保留所有 4 个主成分）对数据进行拟合和变换
# 创建 PCA 对象
# 参数说明：
#   n_components=4: 保留 4 个主成分（等于原始特征数，不进行降维）
#   svd_solver='full': 指定分解方式为完整 SVD 分解（等价于协方差矩阵的特征分解）
# 注意：sklearn 的 PCA 会自动对数据进行中心化（减去均值），但不会缩放（不除以标准差）
pca_full = PCA(n_components=4, svd_solver='full')

# 对原始特征矩阵 X 进行拟合并返回变换后的数据
# fit_transform 方法依次完成：
#   1. 计算原始数据的协方差矩阵
#   2. 进行特征值分解，得到特征向量和特征值
#   3. 将原始数据投影到主成分空间
# 返回的 X_trans 是一个形状为 (150, 4) 的 NumPy 数组，每一列为一条数据的主成分得分
X_trans = pca_full.fit_transform(X)

# 输出第一条数据在主成分分析后的值（即第一行变换后的结果）
print("\n第一条数据主成分分析后值：")
print(X_trans[0])

# =============================================================================
# 4. 计算并输出变换后数据的协方差矩阵
# 使用 numpy 的 cov 函数
# 参数说明：
#   X_trans: 变换后的数据矩阵，每行为一个样本，每列为一个主成分
#   rowvar=False: 指明每一列代表一个变量（默认是每一行代表一个变量）
# 理论上该矩阵应为对角阵（或接近对角阵），对角线上的值为主成分的方差（即特征值）
cov_trans = np.cov(X_trans, rowvar=False)
print("\n变换后数据协方差矩阵：")
print(cov_trans)

# =============================================================================
# 5. 输出每个特征的贡献度（方差解释率），从大到小排序
# 从已拟合的 PCA 对象中获取每个主成分的方差解释率（贡献度）
# explained_variance_ratio_ 属性：各主成分方差占原始总方差的比例，已按从大到小排序
explained_var_ratio = pca_full.explained_variance_ratio_

# 为主成分生成名称，如 PC1, PC2, PC3, PC4
pc_names = [f'PC{i+1}' for i in range(len(explained_var_ratio))]

print("\n每个主成分的贡献度（从大到小排序）：")
# 遍历主成分名称和对应的贡献度，按顺序输出
for name, ratio in zip(pc_names, explained_var_ratio):
    # 输出格式：特征名称    贡献度（保留6位小数）
    print(f"{name}    {ratio:.6f}")

# =============================================================================
# 6. 输出奇异值、特征值、特征向量（均已按从大到小排序）
# 奇异值：PCA 的 singular_values_ 属性
# 奇异值 = sqrt(特征值 * (n_samples - 1))，反映了数据在主成分方向上的散布程度
print("\n奇异值（从大到小）：")
print(pca_full.singular_values_)

# 特征值：PCA 的 explained_variance_ 属性
# 每个特征值代表对应主成分方向上的方差大小，即协方差矩阵的特征值
print("\n特征值（从大到小）：")
print(pca_full.explained_variance_)

# 特征向量：PCA 的 components_ 属性
# 每一行是一个特征向量（主成分的投影方向），行索引对应主成分的顺序（从大到小）
# 每个向量定义了原始特征如何线性组合成该主成分
print("\n特征向量（按特征值从大到小排列，每行一个向量）：")
print(pca_full.components_)

# =============================================================================
# 7. 使用 PCA 进行降维，降到 2 维
# 创建 PCA 对象
# 参数说明：
#   n_components=2: 指定降维后的维度为 2
#   svd_solver='full': 手动选择完整 SVD 分解（协方差矩阵特征分解方式）
#       也可以使用 'auto'、'arpack'、'randomized' 等其他分解方式
pca_2d = PCA(n_components=2, svd_solver='full')

# 对原始特征矩阵 X 进行拟合和降维
# 返回形状为 (150, 2) 的数组，包含每个样本在前两个主成分上的投影
X_reduced = pca_2d.fit_transform(X)

# =============================================================================
# 8. 输出降维后的 2 维数据所代表的原始信息百分比
# 通过 explained_variance_ratio_ 的前两项求和，得到 2 维保留的总方差百分比
info_percentage = pca_2d.explained_variance_ratio_.sum()

print("\n降维到 2 维后保留的原始信息百分比：")
# 将百分比数值乘以 100 并以百分数形式显示，保留两位小数
print(f"{info_percentage * 100:.2f}%")