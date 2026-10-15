# 导入pandas库，用于读取CSV数据集、执行数据预处理与特征工程
# pandas是Python主流表格数据处理工具，支持DataFrame格式的灵活增删改查
import pandas as pd

# 从sklearn.model_selection模块导入train_test_split函数
# 功能：将数据集随机拆分为训练集（用于模型拟合）和测试集（用于泛化能力评估）
# 支持分层抽样、固定随机种子、自定义拆分比例等能力
from sklearn.model_selection import train_test_split

# 从sklearn.ensemble模块导入GradientBoostingClassifier类
# 功能：实现梯度提升决策树（GBDT）分类算法，通过多棵弱回归树串行集成完成分类任务
# sklearn 1.9.0版本完全兼容以下代码中使用的所有参数
from sklearn.ensemble import GradientBoostingClassifier

# 从sklearn.metrics模块导入accuracy_score函数
# 功能：计算分类任务的准确率指标，即预测正确的样本数量占总样本数量的比例
# 输入真实标签数组与预测标签数组，返回0~1之间的浮点数结果
from sklearn.metrics import accuracy_score


# ---------------------- 1. 读取数据集 ----------------------
# 读取本地蘑菇数据集CSV文件，结果存储为DataFrame格式赋值给变量data
# 参数说明：
# filepath_or_buffer: 数据集本地绝对路径，r前缀表示原始字符串，避免Windows路径反斜杠转义问题
# sep=',': 指定文件列分隔符为逗号，符合CSV文件标准格式
data = pd.read_csv(r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv", sep=';')


# ---------------------- 2. 分离特征与标签 ----------------------
# 从原始数据中提取特征矩阵X：删除标签列'class'，剩余所有列作为模型输入特征
# 参数说明：
# labels='class': 指定要删除的列名称
# axis=1: 表示按列维度执行删除操作（axis=0为按行删除）
X = data.drop(labels='class', axis=1)

# 从原始数据中提取标签向量y：'class'列为分类目标，代表蘑菇可食用性
# 标签取值：e = 可食用（edible），p = 有毒（poisonous）
y = data['class']


# ---------------------- 3. 特征编码（数据预处理） ----------------------
# 对特征矩阵执行独热编码（One-Hot Encoding），将离散分类特征转换为数值型
# 原因：GBDT模型无法直接处理字符串类型的分类特征，必须转换为数值格式才可训练
# pd.get_dummies自动识别所有字符串/分类类型列并生成独热编码，数值型列保持原样不变
X_encoded = pd.get_dummies(X)


# ---------------------- 4. 拆分训练集与测试集 ----------------------
# 将编码后的特征与标签拆分为训练集和测试集，按顺序返回4个结果：
# 训练集特征、测试集特征、训练集标签、测试集标签
# 参数说明：
# X_encoded: 输入的编码后特征矩阵
# y: 输入的标签向量
# test_size=0.2: 测试集占总数据集的比例为20%，对应训练集占80%
# random_state=42: 随机数种子，固定随机拆分逻辑，保证每次运行拆分结果一致，提升实验可复现性
# stratify=y: 按标签y进行分层抽样，保证训练集、测试集的正负样本比例与原数据集完全一致
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------- 5. 初始化GBDT模型（手动指定弱分类器） ----------------------
# GBDT的基弱分类器默认为CART回归树，通过以下参数手动指定弱分类器的复杂度与结构
# 参数说明：
# n_estimators=100: 弱分类器（决策树）的总数量，即梯度提升的迭代轮次
# learning_rate=0.1: 学习率（缩减系数），控制每棵弱分类器对最终结果的贡献权重
#   学习率越小，达到同等效果需要的迭代次数越多，通常与n_estimators配合调优
# max_depth=3: 每棵弱决策树的最大深度，手动控制弱分类器的拟合能力
#   深度越小，单棵树的拟合能力越弱（符合弱分类器定义），可有效降低过拟合风险
# min_samples_split=2: 决策树节点分裂所需的最小样本数，节点样本数小于该值则停止分裂
# random_state=42: 随机种子，固定模型训练过程中的随机性，保证结果可复现
gbdt_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    random_state=42
)


# ---------------------- 6. 训练GBDT模型 ----------------------
# 使用训练集数据拟合GBDT模型，完成模型训练
# 参数说明：
# X_train: 训练集特征矩阵，作为模型的输入数据
# y_train: 训练集标签向量，作为模型学习的目标值
# fit方法会执行完整的梯度提升迭代流程，依次生成多棵弱决策树并完成集成
gbdt_model.fit(X_train, y_train)


# ---------------------- 7. 计算训练集与测试集精度 ----------------------
# 对训练集特征执行预测，得到训练集的预测标签数组
# 参数说明：
# X_train: 输入训练集特征矩阵
# 返回值：与输入样本一一对应的预测标签数组
y_train_pred = gbdt_model.predict(X_train)

# 计算训练集分类准确率
# 参数说明：
# y_true=y_train: 训练集真实标签
# y_pred=y_train_pred: 训练集预测标签
# 返回值：准确率数值，范围0~1，值越大代表模型在训练集上拟合效果越好
train_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)

# 对测试集特征执行预测，得到测试集的预测标签数组
y_test_pred = gbdt_model.predict(X_test)

# 计算测试集分类准确率，用于评估模型的泛化能力（在未见过数据上的表现）
test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)

# 打印训练集精度，格式化输出保留4位小数，提升可读性
print(f"训练集分类精度：{train_accuracy:.4f}")
# 打印测试集精度
print(f"测试集分类精度：{test_accuracy:.4f}")


# ---------------------- 8. 单条测试数据预测 ----------------------
# 构造单条测试样本，特征名称与取值严格与原数据集的特征列保持一致
# 示例为一组典型蘑菇特征，可根据需求修改各特征的取值
test_data = pd.DataFrame([{
    "cap-diameter": 5.5,
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "does-bruise-or-bleed": "f",
    "gill-attachment": "a",
    "gill-spacing": "c",
    "gill-color": "k",
    "stem-height": 2.5,
    "stem-width": 1.3,
    "stem-root": "e",
    "stem-surface": "s",
    "stem-color": "w",
    "veil-type": "u",
    "veil-color": "w",
    "has-ring": "t",
    "ring-type": "p",
    "spore-print-color": "k",
    "habitat": "u",
    "season": "u"
}])

# 对测试样本执行与训练集完全一致的独热编码，保证特征处理逻辑统一
test_data_encoded = pd.get_dummies(test_data)

# 对齐测试样本与训练集的特征列，确保特征数量、顺序完全匹配
# 原因：单条测试样本的分类特征取值有限，独热编码后列数会少于训练集
# reindex按训练集的列名重新索引，缺失列用0填充，避免模型预测时因维度不匹配报错
# 参数说明：
# columns=X_encoded.columns: 指定对齐的目标列名（训练集编码后的所有列）
# fill_value=0: 缺失的特征列填充数值0
test_data_encoded = test_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# 使用训练好的GBDT模型对测试样本执行预测
# 输入处理后的测试样本特征，返回预测标签数组（长度为1）
test_pred_result = gbdt_model.predict(test_data_encoded)

# 打印测试数据的原始预测标签
print(f"\n测试数据的预测标签：{test_pred_result[0]}")

# 根据预测标签输出可读性更强的结论说明
if test_pred_result[0] == 'e':
    print("预测结论：该蘑菇为可食用类型（edible）")
else:
    print("预测结论：该蘑菇为有毒类型（poisonous）")
