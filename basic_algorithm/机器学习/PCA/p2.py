# 导入pandas库，用于读取本地数据集并进行表格型数据的读取与预处理
import pandas as pd
# 导入numpy库，用于数值计算、矩阵运算以及协方差、特征值等统计量计算
import numpy as np
# 从sklearn.decomposition模块导入PCA类，用于实现主成分分析降维算法
from sklearn.decomposition import PCA
# 从sklearn.preprocessing模块导入StandardScaler类，用于对特征数据进行标准化处理
from sklearn.preprocessing import StandardScaler

# 定义鸢尾花数据集的本地文件绝对路径
# 使用原始字符串r前缀，避免Windows路径中反斜杠的转义问题
data_path = r"D:\Repositories\basic_algorithm\database\iris.data"

# 使用pandas的read_csv函数读取本地CSV格式数据集
# 参数filepath_or_buffer：传入数据集的本地路径字符串
# 参数header：设置为None，表示数据集文件无表头行，使用默认整数索引作为列名
df = pd.read_csv(filepath_or_buffer=data_path, header=None)

# 通过iloc按位置索引提取特征矩阵，并转为numpy数组
# 第一个维度: 表示选取全部样本行
# 第二个维度0:4：左闭右开区间，选取索引为0、1、2、3的前4列，对应4个原始特征
# .values：将pandas的DataFrame转为numpy数组，方便后续矩阵运算
X = df.iloc[:, 0:4].values
# 提取标签向量：选取所有行的第4列（索引4），对应鸢尾花的类别标签
y = df.iloc[:, 4].values

# ---------------------- 1. 输出原始第一条数据 ----------------------
print("=" * 55)
print("1. 原始第一条数据的特征值")
print("=" * 55)
# X[0]：取特征矩阵的第0行，对应第一条样本的所有原始特征值
print(X[0])
print()

# ---------------------- 2. 输出原始数据协方差矩阵 ----------------------
# 计算原始特征数据的协方差矩阵
# 调用numpy的cov函数计算协方差
# 参数m：传入待计算的特征矩阵X
# 参数rowvar：设置为False，表示每一列代表一个变量（特征），每一行代表一个样本
# 返回值：形状为(特征数, 特征数)的二维协方差矩阵
cov_original = np.cov(m=X, rowvar=False)

print("=" * 55)
print("2. 原始数据协方差矩阵")
print("=" * 55)
print(cov_original)
print()

# ---------------------- 数据标准化（PCA前置处理） ----------------------
# 实例化StandardScaler标准化对象
# 作用：将特征数据转换为均值为0、标准差为1的标准正态分布，消除量纲差异对PCA的影响
# 无必需传入参数，默认使用样本均值和样本标准差进行标准化计算
scaler = StandardScaler()

# 对原始特征矩阵执行拟合+标准化转换
# fit_transform方法：先拟合数据（计算每列特征的均值和标准差），再基于统计量执行标准化转换
# 参数X：传入待处理的原始特征矩阵
# 返回值：标准化后的numpy数组，形状与输入一致
X_scaled = scaler.fit_transform(X)

# ---------------------- PCA模型拟合（SVD分解方式） ----------------------
# 实例化PCA主成分分析模型
# 参数n_components：设置为4，保留全部4个主成分，用于获取完整的分解结果
# 参数svd_solver：设置为'full'，执行完整的奇异值分解(SVD)，保证小数据集下结果精确
# 其余参数使用默认值：whiten=False不进行白化，copy=True不修改原输入数据
pca = PCA(n_components=4, svd_solver='full')

# 用标准化后的数据拟合PCA模型，同时将数据投影到主成分空间
# fit_transform方法：先拟合模型（计算主成分方向、方差、奇异值等核心参数），再将输入数据转换为主成分坐标
# 参数X：传入标准化后的特征矩阵
# 返回值：投影到全部主成分空间后的数据集，形状为(样本数, 主成分数)
X_pca_all = pca.fit_transform(X_scaled)

# ---------------------- 3. 输出第一条数据PCA后的值 ----------------------
print("=" * 55)
print("3. 第一条数据主成分分析后的值（4维）")
print("=" * 55)
# X_pca_all[0]：取变换后数组的第0行，对应第一条样本在4个主成分上的投影坐标
print(X_pca_all[0])
print()

# ---------------------- 4. 输出变换后数据协方差矩阵 ----------------------
# 计算PCA变换后数据的协方差矩阵
# 理论上主成分间互不相关，协方差矩阵应为对角矩阵，对角元素为各主成分的方差（即特征值）
# 参数m：传入PCA变换后的全量数据
# 参数rowvar：设置为False，按列计算变量协方差
cov_pca = np.cov(m=X_pca_all, rowvar=False)

print("=" * 55)
print("4. 变换后数据协方差矩阵")
print("=" * 55)
print(cov_pca)
print()

# ---------------------- 5. 输出原始特征贡献度排序 ----------------------
# 获取PCA的主成分系数矩阵（载荷矩阵）
# components_是PCA拟合后的内置属性，形状为(主成分数, 原始特征数)
# 每行对应一个主成分，每列对应一个原始特征在该主成分中的权重系数（特征向量元素）
components = pca.components_

# 获取每个主成分的方差解释率（贡献度）
# explained_variance_ratio_是PCA内置属性，返回numpy数组，元素按主成分方差从大到小自动排序
explained_var_ratio = pca.explained_variance_ratio_

# 定义鸢尾花数据集4个原始特征的名称，用于结果展示
feature_names = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]

# 计算每个原始特征的总贡献度
# 计算逻辑：以主成分的方差解释率为权重，对特征在各主成分的系数绝对值进行加权求和
# np.abs(components)：取系数绝对值，消除正负号影响，仅衡量贡献大小
# explained_var_ratio.reshape(-1, 1)：将一维方差比率数组转为列向量，通过广播机制与系数矩阵对应相乘
# np.sum(axis=0)：按列求和，得到每个原始特征的总贡献得分
feature_contribution = np.sum(np.abs(components) * explained_var_ratio.reshape(-1, 1), axis=0)

# 将特征名称与对应贡献度一一配对，转换为字典结构
# zip函数：将两个列表按位置一一对应组合
feature_contrib_dict = dict(zip(feature_names, feature_contribution))

# 按特征贡献度从大到小进行排序
# sorted函数：对字典的items()进行排序
# key参数：指定排序依据为元组的第二个元素（贡献度数值）
# reverse=True：表示降序排列（从大到小）
sorted_feature_contrib = sorted(feature_contrib_dict.items(), key=lambda x: x[1], reverse=True)

print("=" * 55)
print("5. 原始特征贡献度排序（从大到小）")
print("=" * 55)
# 格式化输出表头，左对齐保证排版整齐
print(f"{'特征名称':<12}{'贡献度':<10}")
# 遍历排序后的结果，逐行输出特征名称与贡献度（保留4位小数）
for name, contrib in sorted_feature_contrib:
    print(f"{name:<14}{contrib:.4f}")
print()

# ---------------------- 6. 输出奇异值、特征值、特征向量 ----------------------
# 获取PCA奇异值分解得到的奇异值，按从大到小自动排序
# singular_values_是sklearn 1.9.0稳定支持的PCA内置属性，对应每个主成分的奇异值
singular_values = pca.singular_values_

# 获取协方差矩阵的特征值，即每个主成分的解释方差，按从大到小排序
# explained_variance_是PCA内置属性，对应标准化后数据协方差矩阵的特征值
eigenvalues = pca.explained_variance_

# 获取协方差矩阵的特征向量，每行对应一个主成分的特征向量
# 行顺序与特征值从大到小的顺序一一对应
eigenvectors = pca.components_

# 打印奇异值
print("=" * 55)
print("6. 奇异值（从大到小排序）")
print("=" * 55)
for idx, val in enumerate(singular_values):
    print(f"奇异值{idx+1}: {val:.4f}")
print()

# 打印特征值
print("=" * 55)
print("7. 特征值（从大到小排序）")
print("=" * 55)
for idx, val in enumerate(eigenvalues):
    print(f"特征值{idx+1}: {val:.4f}")
print()

# 打印特征向量
print("=" * 55)
print("8. 特征向量（按对应特征值从大到小排序，每行一个）")
print("=" * 55)
for idx, vec in enumerate(eigenvectors):
    # np.round：对数组元素保留4位小数，提升可读性
    print(f"特征向量{idx+1}: {np.round(vec, 4)}")
print()

# ---------------------- 7. 降维到2维 + 信息保留占比 ----------------------
# 提取前2个主成分的投影数据，实现从4维降维到2维
# 切片[:, :2]：选取所有行的前2列，对应方差最大的两个主成分
X_pca_2d = X_pca_all[:, :2]

# 计算前2个主成分的累计方差解释率，即保留的原始数据信息百分比
# explained_var_ratio[:2]：取前2个主成分的方差解释率
# np.sum求和后乘以100，转换为百分比格式
cumulative_variance_2d = np.sum(explained_var_ratio[:2]) * 100

print("=" * 55)
print("9. 降维到2维保留的原始信息百分比")
print("=" * 55)
print(f"前2个主成分累计保留信息占比：{cumulative_variance_2d:.2f}%")
print()

# 打印降维后的数据形状，验证维度结果
print(f"降维到2维后的数据形状（样本数, 维度数）：{X_pca_2d.shape}")
