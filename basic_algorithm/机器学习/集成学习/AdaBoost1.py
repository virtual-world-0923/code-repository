# 导入pandas库，用于数据读取、数据框操作与数据预处理
import pandas as pd
# 导入numpy库，用于数值计算与数组格式处理
import numpy as np
# 导入train_test_split函数，用于将数据集拆分为训练集与测试集
from sklearn.model_selection import train_test_split
# 导入AdaBoostClassifier类，实现AdaBoost集成学习分类算法（适配sklearn 1.9.0 官方API）
from sklearn.ensemble import AdaBoostClassifier
# 导入DecisionTreeClassifier类，作为AdaBoost的基弱分类器
from sklearn.tree import DecisionTreeClassifier
# 导入OrdinalEncoder类，用于将分类特征转换为数值型序数编码
from sklearn.preprocessing import OrdinalEncoder
# 导入LabelEncoder类，用于将目标标签转换为数值编码
from sklearn.preprocessing import LabelEncoder
# 导入SimpleImputer类，用于填充数据中的缺失值（适配蘑菇数据集的缺失字段）
from sklearn.impute import SimpleImputer
# 导入ColumnTransformer类，对不同类型的特征列分别应用不同的预处理逻辑
from sklearn.compose import ColumnTransformer
# 导入Pipeline类，用于构建多步骤的特征处理流水线
from sklearn.pipeline import Pipeline


# ---------------------- 1. 读取数据集（分号分隔符适配） ----------------------
# 读取本地蘑菇数据集CSV文件，该数据集使用分号作为列分隔符
# 参数说明：
#   filepath_or_buffer：数据集文件的本地路径，前缀r表示原始字符串，避免Windows路径反斜杠转义问题
#   sep=';'：指定列分隔符为分号，匹配secondary_data.csv的真实格式
#   header=0：指定第0行（第一行）作为列名
df = pd.read_csv(
    r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv",
    sep=';',
    header=0
)

# print(f"\n各列数据类型:\n{df.dtypes}")

# ---------------------- 2. 分离特征与标签 ----------------------
# 提取特征矩阵X：删除目标标签列'class'，剩余所有列作为模型输入特征
# 参数说明：
#   labels='class'：指定要删除的列名
#   axis=1：表示按列进行删除操作（axis=0为按行删除）
X = df.drop(labels='class', axis=1)
# 提取目标标签向量y：取'class'列作为分类预测的目标变量（e=可食用，p=有毒）
y = df['class']


# ---------------------- 3. 拆分训练集与测试集 ----------------------
# 将原始数据集按比例拆分为训练集和测试集
# 参数说明：
#   X：输入特征矩阵
#   y：输入目标标签向量
#   test_size=0.2：测试集占总样本的比例为20%，训练集占80%
#   random_state=42：随机数种子，固定随机拆分结果，保证实验可复现
#   stratify=y：按标签y的类别比例进行分层抽样，保证训练集与测试集的类别分布一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------- 4. 特征预处理（缺失值填充+分类编码） ----------------------
# 识别训练集中的分类特征列：筛选字符串类型的列
# 参数说明：
#   include=['object', 'string']：指定筛选数据类型为object或string的列，columns.tolist()：将筛选出的列名转换为列表格式
categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
# 识别训练集中的数值特征列：排除所有字符串类型的列，剩余为数值列
# 参数说明：
#   exclude=['object', 'string']：指定排除所有字符串类型的列，剩余int、float等为数值特征
numerical_cols = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()

# 构建分类特征处理流水线：依次执行缺失值填充、序数编码
# Pipeline参数说明：
#   steps：流水线步骤列表，每个元素为(步骤名称, 处理器实例)
cat_pipeline = Pipeline(steps=[
    # 第一步：用众数填充分类特征的缺失值
    # SimpleImputer参数说明：
    #   strategy='most_frequent'：填充策略为使用出现频率最高的值（众数）
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # 第二步：对特征进行序数编码，将字符串类别转换为数值
    # OrdinalEncoder参数说明：
    #   handle_unknown='use_encoded_value'：遇到训练集中未出现的未知类别时，使用指定值编码
    #   unknown_value=-1：将未知类别编码为-1，避免预测过程中报错
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 构建数值特征处理流水线：执行缺失值填充
num_pipeline = Pipeline(steps=[
    # 用中位数填充数值特征的缺失值
    # SimpleImputer参数说明：
    #   strategy='median'：填充策略为使用特征的中位数
    ('imputer', SimpleImputer(strategy='median'))
])

# 构建完整的特征预处理转换器，对分类特征和数值特征分别应用对应流水线
# ColumnTransformer参数说明：
#   transformers：转换器列表，每个元素为(名称, 转换器, 处理的列列表)
#   ('cat', cat_pipeline, categorical_cols)：对分类列应用分类处理流水线
#   ('num', num_pipeline, numerical_cols)：对数值列应用数值处理流水线
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipeline, categorical_cols),
    ('num', num_pipeline, numerical_cols)
])

# 在训练集上拟合预处理流水线（学习填充规则、编码规则），并转换得到预处理后的训练集特征
# fit_transform方法：先拟合数据学习参数，再对数据进行转换
X_train_encoded = preprocessor.fit_transform(X_train)
# 使用训练集拟合好的预处理流水线转换测试集特征，禁止在测试集上拟合，避免数据泄露
# transform方法：使用已学习的参数对新数据进行转换
X_test_encoded = preprocessor.transform(X_test)


# ---------------------- 5. 目标标签编码 ----------------------
# 创建标签编码器实例，用于将字符串类型的目标标签转换为数值型标签
le = LabelEncoder()
# 在训练集标签上拟合编码器（学习标签与数值的映射关系），并转换训练集标签
y_train_encoded = le.fit_transform(y_train)
# 使用训练好的编码器转换测试集标签
y_test_encoded = le.transform(y_test)


# ---------------------- 6. 手动指定弱分类器 + 构建AdaBoost模型 ----------------------
# 手动指定AdaBoost的基弱分类器：最大深度为1的决策树（又称决策树桩，AdaBoost最常用的基分类器）
# DecisionTreeClassifier参数说明：
#   max_depth=1：决策树的最大深度为1，仅包含一个分裂节点，保证模型为弱分类器
#   random_state=42：决策树的随机种子，固定分裂结果，保证实验可复现
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# 创建AdaBoost分类器实例
# 参数说明：
#   estimator=base_estimator：指定基弱分类器（1.9.0版本使用estimator参数，旧版base_estimator已弃用）
#   n_estimators=50：集成的弱分类器总数量，即算法迭代的最大次数
#   learning_rate=1.0：学习率，用于缩放每个弱分类器的权重贡献，取值越小需要的弱分类器越多
#   random_state=42：随机种子，固定样本权重初始化与弱分类器训练的随机性，保证结果可复现
# 注：sklearn 1.9.0已移除algorithm参数，默认使用SAMME.R概率提升算法，无需手动指定
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)


# ---------------------- 7. 训练模型 ----------------------
# 使用预处理后的训练集数据训练AdaBoost模型
# fit方法参数说明：
#   X：训练集特征矩阵
#   y：训练集目标标签
ada_model.fit(X_train_encoded, y_train_encoded)


# ---------------------- 8. 输出训练集与测试集精度 ----------------------
# 计算模型在训练集上的分类准确率
# score方法：返回预测正确样本数占总样本数的比例，即准确率
train_accuracy = ada_model.score(X_train_encoded, y_train_encoded)
# 计算模型在测试集上的分类准确率
test_accuracy = ada_model.score(X_test_encoded, y_test_encoded)

# 格式化打印训练集精度，保留4位小数
print(f"训练集分类精度: {train_accuracy:.4f}")
# 格式化打印测试集精度
print(f"测试集分类精度: {test_accuracy:.4f}")


# ---------------------- 9. 输入测试数据并打印预测结果 ----------------------
# 构造单条测试样本，列名与原始特征X的列名完全一致，保证预处理时列匹配
# 特征顺序与secondary_data.csv的特征列顺序完全对应
test_sample = pd.DataFrame(
    data=[[5.2, 'x', 's', 'n', 'f', 'a', 'c', 'k', 6.1, 2.3, 'e', 's', 'w', 'u', 'w', 't', 'p', 'k', 'u', 'a']],
    columns=X.columns  # 列名与原始特征列名完全对齐，避免顺序错误
)

# 使用训练好的预处理流水线对测试样本进行特征编码与预处理
test_sample_encoded = preprocessor.transform(test_sample)

# 使用训练好的AdaBoost模型对测试样本进行预测，返回编码后的预测标签
# predict方法：返回输入样本的预测类别
pred_encoded = ada_model.predict(test_sample_encoded)
# 将数值编码的预测结果转换回原始的字符串标签
# inverse_transform方法：将数值编码还原为原始类别名称
pred_label = le.inverse_transform(pred_encoded)

# 打印测试样本的预测结果
print("\n测试数据预测结果:")
print(f"预测类别编码: {pred_encoded[0]}")
print(f"预测类别名称: {pred_label[0]}")
# 补充类别含义说明
if pred_label[0] == 'e':
    print("类别含义：该蘑菇预测为可食用（edible）")
else:
    print("类别含义：该蘑菇预测为有毒（poisonous）")
