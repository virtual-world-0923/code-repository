# -*- coding: utf-8 -*-
# 导入必要的库

import pandas as pd  # 导入 pandas 库，用于数据处理，提供 DataFrame 结构
import numpy as np   # 导入 numpy 库，用于科学计算，提供数组操作和 NaN 处理
from sklearn.model_selection import train_test_split  # 导入数据集拆分函数，用于划分训练集和测试集
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder  # 导入编码器，用于将类别特征和标签转换为数值
from sklearn.impute import SimpleImputer  # 导入缺失值填充器，用于处理数据中的缺失值
from sklearn.ensemble import GradientBoostingClassifier  # 导入 GBDT 分类器，实现梯度提升树算法

# ==================== 1. 加载数据集 ====================
# 数据集路径，使用原始字符串避免转义问题
data_path = r'D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv'
# 使用 pandas 读取 CSV 文件，返回 DataFrame 对象
df = pd.read_csv(data_path,sep=";")

# ==================== 2. 数据预处理 ====================
# 将数据集中的 '?' 替换为 numpy 的 NaN，便于后续缺失值处理
df.replace('?', np.nan, inplace=True)  # to_replace='?', value=np.nan, inplace=True 表示在原 DataFrame 上修改

# 分离特征与目标变量
# 假设数据集中目标列名为 'class'，若实际不同请对应修改
target_column = 'class'
X = df.drop(target_column, axis=1)  # 删除目标列，保留所有特征列，axis=1 表示按列删除
y = df[target_column]               # 提取目标列，类型为 pandas Series

# ---- 处理特征缺失值 ----
# 创建缺失值填充器，使用众数（最频繁出现的值）填充
imputer = SimpleImputer(strategy='most_frequent')
# 参数: strategy='most_frequent' 表示用每列出现次数最多的值填充缺失位置
X_imputed = imputer.fit_transform(X)  # 返回填充后的 numpy 数组，形状与原数据相同

# ---- 编码特征 ----
# 创建有序编码器，将类别字符串转换为整数，并设置对未知类别的处理方式
# encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.bool_)
X_encoded = encoder.fit_transform(X_imputed)  # 拟合并转换特征，返回编码后的 numpy 数组
print("编码后特征维度:", X_encoded.shape)

# ---- 编码目标变量 ----
le = LabelEncoder()  # 创建标签编码器，用于将目标类别转为 0/1 等整数
y_encoded = le.fit_transform(y)  # 拟合并转换标签，返回一维 numpy 数组

# ==================== 3. 拆分训练集与测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2,        # 参数: 测试集占总数据的比例，这里设为 20%
    random_state=42,      # 随机种子，保证结果可复现
    stratify=y_encoded    # 按照标签分布进行分层抽样，使训练/测试集类别比例与原数据一致
)

# ==================== 4. 手动指定弱分类器，创建 GBDT 模型 ====================
# 使用 GradientBoostingClassifier，通过参数手动将弱学习器指定为深度为1的决策树（决策树桩）
gbdt = GradientBoostingClassifier(
    n_estimators=100,      # 弱分类器（决策树）的数量，即迭代轮数
    learning_rate=0.1,     # 学习率，控制每个弱分类器对最终结果的贡献大小
    max_depth=3, # 弱分类器（决策树）的最大深度，设置为1即为树桩（单一分裂点）
    verbose=1, #日志级别
    random_state=42        # 随机种子，保证训练过程可复现
)
# 注意：默认的基础学习器是决策树回归器，通过 max_depth、min_samples_split 等参数可“手动指定”弱分类器的结构

# ==================== 5. 训练模型 ====================
gbdt.fit(X_train, y_train)  # 使用训练集特征和标签拟合模型

# ==================== 6. 评估并输出训练集和测试集精度 ====================
train_accuracy = gbdt.score(X_train, y_train)  # 计算模型在训练集上的准确率，返回浮点数
test_accuracy = gbdt.score(X_test, y_test)     # 计算模型在测试集上的准确率
print(f"训练集精度: {train_accuracy:.4f}")     # 输出训练集精度，保留4位小数
print(f"测试集精度: {test_accuracy:.4f}")     # 输出测试集精度

# ==================== 7. 输入测试数据，打印预测结果 ====================
# 选取测试集前5个样本作为示例测试数据
sample_X = X_test[:5]   # 取前5行特征
sample_y = y_test[:5]   # 对应的真实标签

# 使用训练好的模型进行预测
y_pred = gbdt.predict(sample_X)  # 返回预测的类别标签（整数编码）

# 将编码后的标签转换回原始类别字符串，便于理解
y_true_labels = le.inverse_transform(sample_y)  # 真实标签解码
y_pred_labels = le.inverse_transform(y_pred)    # 预测标签解码

# 打印每个样本的真实类别与预测类别
print("\n测试集样本预测结果:")
for i in range(len(y_true_labels)):
    print(f"样本{i+1}: 真实类别 = {y_true_labels[i]}, 预测类别 = {y_pred_labels[i]}")

# 额外输出预测概率（可选）
y_proba = gbdt.predict_proba(sample_X)  # 返回每个样本属于各类别的概率，形状 (n_samples, n_classes)
class_names = le.classes_               # 获取原始的类别名称列表
print("\n预测概率（前5个样本）:")
for i, proba in enumerate(y_proba):
    # 将概率与类别名称组合展示
    proba_str = ", ".join([f"{cls}: {p:.3f}" for cls, p in zip(class_names, proba)])
    print(f"样本{i+1}: {proba_str}")