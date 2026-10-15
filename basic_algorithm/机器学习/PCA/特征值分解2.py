import numpy as np   # 导入NumPy库，版本需为2.5.1，用于数值计算与线性代数操作
import pandas as pd  # 导入Pandas库，版本需为3.0.3，用于数据读取与处理

# 1. 读取鸢尾花数据集
# 数据集文件路径，使用原始字符串避免转义字符问题
# 该文件没有列名，前4列为特征，第5列为类别标签
data_path = r"D:\Repositories\basic_algorithm\database\iris.data"

# 定义列名，方便后续提取特征列
# 列名依次为：花萼长度、花萼宽度、花瓣长度、花瓣宽度、类别
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 使用pandas读取CSV文件
# 参数 header=None 表示文件第一行不是列名，names 参数用于手动指定列名
df = pd.read_csv(data_path, header=None, names=column_names)

# 2. 提取特征矩阵（只取前4列，不包含类别列）
# iloc[:, :-1] 选择所有行，列从开始到倒数第二列（即排除最后一列class）
# .values 将DataFrame转换为NumPy二维数组，形状为 (150, 4)
X = df.iloc[:, :-1].values

# 3. 数据中心化（去均值），使每个特征列的均值为0
# np.mean(X, axis=0) 计算每个特征列的均值，axis=0表示沿行方向求均值（对每列操作）
# 返回一个形状为 (4,) 的一维数组，包含4个特征的均值
mean_vec = np.mean(X, axis=0)

# 将原始数据矩阵减去对应列的均值，实现中心化
# 广播机制会自动将mean_vec扩展为与X相同的形状进行逐元素相减
X_centered = X - mean_vec

# 4. 计算中心化数据的协方差矩阵
# np.cov 计算协方差矩阵，参数 rowvar=False 表示每一列代表一个变量（特征），行代表观测样本
# 协方差矩阵 shape 为 (4, 4)，元素(i, j)为特征i与特征j的协方差
# 分母为 N-1（样本协方差的无偏估计），与PCA中协方差矩阵的计算方式一致
cov_mat = np.cov(X_centered, rowvar=False)

# 5. 对协方差矩阵进行特征值分解
# 由于协方差矩阵是实对称矩阵，使用 np.linalg.eigh 进行特征值分解，效率更高且数值更稳定
# 返回两个值：
#   eigenvalues：一维数组，形状为 (4,)，包含所有特征值，默认按升序排列
#   eigenvectors：二维数组，形状为 (4, 4)，每一列是对应特征值的特征向量
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# 6. 将特征值按从大到小排序，并对应调整特征向量的顺序
# np.argsort(eigenvalues) 返回特征值升序排列的索引数组
# [::-1] 进行切片反转，获得降序排列的索引
sorted_indices = np.argsort(eigenvalues)[::-1]

# 根据降序索引重新排列特征值
eigenvalues_sorted = eigenvalues[sorted_indices]

# 根据降序索引重新排列特征向量矩阵的列（每列是一个特征向量）
# eigenvectors[:, sorted_indices] 表示选取所有行，列的顺序按 sorted_indices 重排
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# 7. 构建投影矩阵，选取前2个主成分（特征值最大的2个特征向量）
# eigenvectors_sorted[:, :2] 取出前两列，形状为 (4, 2)
# 这2个特征向量张成新的2维子空间，数据将被投影到该空间上
W = eigenvectors_sorted[:, :2]

# 8. 将中心化后的原始数据降维到2维
# 使用矩阵乘法进行投影：X_centered 形状 (150, 4) 点乘 W 形状 (4, 2) 得到形状 (150, 2) 的降维数据
# 每一行对应原始数据在新的2维主成分空间中的坐标
X_pca = X_centered.dot(W)

# ---------- 输出结果 ----------
# 输出原始数据集的第一条记录（前4个特征值）
# X[0] 为原始未处理的第一个样本，包含花萼长度、花萼宽度、花瓣长度、花瓣宽度
print("原始数据第一条（特征值）:\n", X[0])

# 输出降维后的第一条记录（在新的2维空间中的坐标）
# X_pca[0] 是第一个样本降维后的两个坐标值
print("\n降维后的第一条数据值:\n", X_pca[0])

# 输出中心化后的协方差矩阵（即特征值分解的原始矩阵）
print("\n协方差矩阵（分解前的矩阵）:\n", cov_mat)

# 输出降序排列的特征值
# 每个特征值表示对应主成分方向上的方差大小
print("\n特征值（降序排列）:\n", eigenvalues_sorted)

# 输出降序排列的特征向量矩阵（每一列是一个主成分方向）
# 该矩阵的每一列与上述特征值一一对应
print("\n特征向量矩阵（主成分方向，每一列为一个特征向量）:\n", eigenvectors_sorted)

# 提示：若需要保留更多小数位数或关闭科学计数，可添加：
# np.set_printoptions(suppress=True, precision=6)