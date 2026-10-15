# -*- coding: utf-8 -*-
"""
使用 XGBoost 对蘑菇数据集（Mushroom Dataset）进行食用/有毒分类练习
严格适配 sklearn 1.9.0 版本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.metrics import accuracy_score            # 用于计算分类准确率
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder  # 用于将类别标签转换为数值
import xgboost as xgb                                 # XGBoost 梯度提升库

# ======================= 1. 加载数据 =======================
# 数据集路径（注意使用原始字符串防止转义）
data_path = r'D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv'

# 读取CSV文件，'?' 在数据集中代表缺失值，通过 na_values 参数将其读取为 NaN
# 参数说明：
#   data_path: 文件路径字符串
#   na_values='?': 指定将数据中的 '?' 识别为缺失值（NaN）
df = pd.read_csv(data_path, sep=";", na_values='?')

# ======================= 2. 数据预处理 =======================
# 该数据集的典型第一列名为 'class'，包含 'e' (edible 可食用) 和 'p' (poisonous 有毒)
# 此处取第一列作为目标列，使其具有更好的通用性
target_column = df.columns[0]   # 取第一列的列名作为目标变量列名

# 分离特征矩阵 X 和目标向量 y
# df.drop(columns=[...]) 返回移除指定列后的新 DataFrame
X = df.drop(columns=[target_column])  # 特征数据，DataFrame 形状 (n_samples, n_features)
y = df[target_column]                 # 目标标签，Series 形状 (n_samples,)

# ---- 对目标标签进行数值编码 ----
# LabelEncoder 将类别字符串转换为从0开始的整数，例如 {'e':0, 'p':1}
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 编码后的标签数组，形状 (n_samples,)，数据类型 int

# ---- 对特征进行编码，同时保留缺失值 NaN，以供 XGBoost 原生缺失值处理 ----
# 遍历每一列特征，将类别型文本特征转换为数值，并保留 NaN
#简易版的 OrdinalEncoder
for col in X.columns:                        # col 是当前列名
    if X[col].dtype == 'object' or 'str':   # 如果该列的数据类型为对象或字符串
        unique_vals = X[col].dropna().unique()  #取出这一列，去掉缺失值（NaN），拿到所有不重复的唯一值
        mapping = {val: i for i, val in enumerate(unique_vals)} # 构建映射字典：{原始类别字符串: 整数编码}
        X[col] = X[col].map(mapping).astype(float) #把列里的每个值按 mapping 字典替换成对应的数字，再转成float
    else:
        # 如果该列已经是数值类型，直接转换为 float（有些可能是 int，统一为 float）
        X[col] = X[col].astype(float)

# encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# X = encoder.fit_transform(X)  # 拟合并转换特征，返回编码后的 numpy 数组

# 经过上述处理，X 中的所有列均为 float 类型，缺失值仍为 NaN
# y_encoded 为 int 类型的 0/1 编码

# ======================= 3. 划分训练集与测试集 =======================
# train_test_split 按指定比例随机划分数据
# 参数说明：
#   X: 特征矩阵
#   y_encoded: 目标向量
#   test_size=0.2: 测试集占总数据的 20%，训练集占 80%
#   random_state=42: 随机种子，保证每次运行划分结果一致
#   stratify=y_encoded: 依据标签分布进行分层抽样，确保训练/测试集中类别比例与原数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ======================= 4. 定义并训练 XGBoost 模型 =======================
# 手动指定弱分类器：通过设置 max_depth=1（决策树桩），使每个基学习器成为极弱的分类器，
# 然后依靠 boosting 集成多个弱分类器形成一个强分类器。
# XGBClassifier 参数说明：
#   n_estimators=100:  使用 100 个弱分类器（提升迭代次数）
#   max_depth=1:        每棵树的最大深度为1，即决策树桩，这是“弱分类器”的核心体现
#   learning_rate=0.1:  学习率，控制每棵树对最终结果的贡献大小，较小的值通常需要更多的树
#   random_state=42:    随机种子，保证结果可复现
#   eval_metric='logloss': 评估指标设为对数损失（用于二分类）
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,            # 关键：手动指定弱分类器（深度为1的决策树桩）
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

# 训练模型，XGBoost 会自动处理特征中的 NaN 值，根据梯度方向决定缺失值的最优默认分裂方向
model.fit(X_train, y_train)

# ======================= 5. 评估模型并输出精度 =======================
# 对训练集进行预测，用于观察模型对已学习数据的拟合程度
y_train_pred = model.predict(X_train)  # 返回训练集预测标签数组，形状 (n_train_samples,)

# 对测试集进行预测，用于评估模型的泛化能力
y_test_pred = model.predict(X_test)    # 返回测试集预测标签数组，形状 (n_test_samples,)

# 计算训练集准确率
# accuracy_score 计算预测正确的样本数占总样本数的比例
# 参数：y_true（真实标签）, y_pred（预测标签）
train_accuracy = accuracy_score(y_train, y_train_pred)

# 计算测试集准确率
test_accuracy = accuracy_score(y_test, y_test_pred)

# 打印训练集和测试集精度，格式化为四位小数
print(f"训练集精度: {train_accuracy:.4f}")
print(f"测试集精度: {test_accuracy:.4f}")

# ======================= 6. 输入测试数据并打印预测结果 =======================
# 此处模拟“输入测试数据”的场景：从测试集中取前5条样本，作为输入数据进行预测并展示结果
print("\n预测测试集中前5个样本的结果：")
sample_indices = [0, 1, 2, 3, 4]  # 选定要展示的测试集样本索引（局部位置索引）

for idx in sample_indices:
    # 提取单个样本，按整数位置索引
    # 需要将一维样本 reshape 为二维数组以符合 predict 输入要求 (1, n_features)
    sample = X_test.iloc[idx].values.reshape(1, -1)

    # 获取该样本的真实标签（已编码的数值）
    true_label = y_test[idx]

    # 调用模型对单个样本进行预测，返回预测的数值标签
    pred_label = model.predict(sample)[0]  # [0] 取出数组中的唯一元素

    # 通过 LabelEncoder 的 inverse_transform 将数值标签还原为原始类别字符串 (e 或 p)
    true_class = le.inverse_transform([true_label])[0]
    pred_class = le.inverse_transform([pred_label])[0]

    # 打印样本编号、真实类别和预测类别
    print(f"样本 {idx}: 真实类别={true_class}, 预测类别={pred_class}")