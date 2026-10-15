import pandas as pd  # 数据处理与读取 CSV
import numpy as np   # 数值计算
from sklearn.model_selection import train_test_split  # 拆分训练集/测试集
from sklearn.tree import DecisionTreeClassifier       # 决策树弱分类器
from sklearn.ensemble import AdaBoostClassifier       # AdaBoost 集成模型
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder  # 类别编码
from sklearn.impute import SimpleImputer               # 缺失值填充
from sklearn.metrics import accuracy_score            # 准确率评估

# ========================== 1. 读取数据 ==========================
# 参数说明：
#   filepath_or_buffer: 文件路径（原始字符串避免转义）
#   sep=';': 指定分隔符为分号（该数据集的标准格式）
data = pd.read_csv(
    r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv",
    sep=';'  # 关键修正点：按分号分隔
)

# 查看数据基本信息，验证读取是否正确
print("数据集形状:", data.shape)
print("前5行数据:\n", data.head())

# ========================== 2. 特征与标签分离 ==========================
# 第一列是标签，其它是特征
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# ========================== 3. 处理缺失值 ==========================
# 由于数据集均为类别字符串，使用众数填充最合适。
# SimpleImputer 参数：
#   missing_values: 需填充的缺失值标记，默认 np.nan
#   strategy: 填充策略，'most_frequent' 表示用出现频率最高的值（众数）填充
imputer = SimpleImputer(strategy='most_frequent')

# fit_transform: 先学习每列的众数，再对 X 进行填充，返回 NumPy 数组
X_imputed = imputer.fit_transform(X)

# ========================== 4. 特征编码（字符串 -> 整数） ==========================
# OrdinalEncoder: 将每个特征列独立编码为整数值（0, 1, 2 ...）
ordinal_encoder = OrdinalEncoder()

# fit_transform: 拟合编码器并对特征进行转换
X_encoded = ordinal_encoder.fit_transform(X_imputed)

# LabelEncoder: 将标签列编码为整数（例如 'e'->0, 'p'->1）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========================== 5. 拆分训练集与测试集 ==========================
# train_test_split 参数：
#   *arrays: 要划分的数组（特征和标签）
#   test_size: 测试集所占比例，0.2 表示 20%
#   random_state: 随机种子，保证划分可复现
#   stratify: 按照标签分布分层抽样，确保训练集/测试集中各类别比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ========================== 6. 手动指定弱分类器 ==========================
# 使用深度为1的决策树桩（Decision Stump）作为 AdaBoost 的基学习器
# DecisionTreeClassifier 参数：
#   max_depth: 树的最大深度，1 表示决策树桩
#   random_state: 随机种子，控制分裂点的随机性
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# ========================== 7. 创建 AdaBoost 分类器 ==========================
# 在 sklearn 1.9.0 中，AdaBoostClassifier 的 algorithm 参数已被移除（仅保留 SAMME 算法），
# 因此只需指定 estimator、n_estimators 和 learning_rate 等参数。
# 参数说明：
#   estimator: 弱学习器（基分类器）
#   n_estimators: 弱分类器的最大数量（迭代次数），默认为 50
#   learning_rate: 学习率/缩减系数，缩小每个弱分类器的贡献，默认为 1.0
#   random_state: 随机种子，控制 boosting 过程中的随机性
adaboost = AdaBoostClassifier(
    estimator=base_estimator,   # 手动指定的弱分类器
    n_estimators=50,            # 弱分类器个数
    learning_rate=1.0,          # 学习率
    random_state=42             # 随机种子
)

# ========================== 8. 训练模型 ==========================
# fit: 用训练数据训练 AdaBoost 模型
# 参数：X 训练特征矩阵，y 训练标签向量
adaboost.fit(X_train, y_train)

# ========================== 9. 模型评估：训练集与测试集精度 ==========================
# predict: 对输入数据进行分类预测，返回预测标签数组
y_train_pred = adaboost.predict(X_train)  # 训练集预测
y_test_pred = adaboost.predict(X_test)    # 测试集预测

# accuracy_score: 计算分类准确率
# 参数：y_true 真实标签，y_pred 预测标签
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n训练集精度：{train_acc:.4f}")
print(f"测试集精度：{test_acc:.4f}")

# ========================== 10. 测试数据预测演示 ==========================
# 从测试集中选取前 5 个样本进行展示
n_demo = 5
X_demo = X_test[:n_demo]      # 取前5个样本的特征
y_demo_true = y_test[:n_demo] # 对应的真实标签

# 对演示样本进行预测
y_demo_pred = adaboost.predict(X_demo)

# inverse_transform: 将整数编码反向映射回原始类别字符串，方便阅读
y_demo_true_label = label_encoder.inverse_transform(y_demo_true)
y_demo_pred_label = label_encoder.inverse_transform(y_demo_pred)

print("\n测试数据预测演示（前5个样本）：")
for i in range(n_demo):
    print(f"样本{i+1}: 真实标签 = {y_demo_true_label[i]}, 预测标签 = {y_demo_pred_label[i]}")

# ------------------ 单一样本预测示例 ------------------
# 选取测试集第一个样本，reshape 为二维数组（predict 要求输入形状为 (n_samples, n_features)）
single_sample = X_test[0].reshape(1, -1)   # reshape(1, -1) 将一维数组转为 1 行多列的二维数组
single_true = y_test[0]
single_pred = adaboost.predict(single_sample)

# 转换为原始标签名称
single_true_label = label_encoder.inverse_transform([single_true])[0]
single_pred_label = label_encoder.inverse_transform(single_pred)[0]
print(f"\n单样本预测：真实 = {single_true_label}, 预测 = {single_pred_label}")