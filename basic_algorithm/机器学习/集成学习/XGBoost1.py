# ============================================================
# 导入数据处理与模型相关库
# ============================================================

# 导入 pandas 库，用于读取 CSV 文件和数据框操作
# pandas 是 Python 中最常用的结构化数据处理库
import pandas as pd

# 导入 numpy 库，用于数组运算和数据格式转换
import numpy as np

# 从 sklearn.model_selection 模块导入 train_test_split 函数
# 作用：将数据集随机划分为训练集和测试集
# 参数说明：支持 test_size、random_state、shuffle、stratify 等
from sklearn.model_selection import train_test_split

# 从 sklearn.preprocessing 模块导入 LabelEncoder
# 作用：将字符串形式的类别标签转换为数值（0, 1, 2...）
# 注意：sklearn 1.9.0 完全支持该类，API 无变更
from sklearn.preprocessing import LabelEncoder

# 从 sklearn.metrics 模块导入 accuracy_score 函数
# 作用：计算预测结果与真实标签的分类准确率
# 公式：准确率 = 预测正确样本数 / 总样本数
from sklearn.metrics import accuracy_score

# 从 xgboost 库导入 XGBClassifier 分类器类
# 作用：XGBoost 梯度提升树分类模型的 sklearn 风格接口
# 关键参数 booster 可手动指定弱分类器类型：gbtree / gblinear / dart
from xgboost import XGBClassifier

# ============================================================
# 1. 读取蘑菇数据集
# ============================================================

# 使用 pandas 的 read_csv 函数读取 CSV 格式的蘑菇数据集
# 参数 filepath_or_buffer：数据集文件的完整路径字符串
# 参数 sep：分隔符，CSV 默认逗号分隔，可省略
# 返回值：DataFrame 格式的完整数据表
data = pd.read_csv(r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv", sep=";")

# ============================================================
# 2. 数据预处理：分离特征与标签，处理分类变量
# ============================================================

# 分离特征矩阵 X 和目标向量 y
# drop 函数：删除指定列，axis=1 表示按列删除
# 参数 labels：要删除的列名列表，此处删除目标列 'class'
# 参数 axis：0 为行，1 为列
# 剩余所有列作为输入特征 X
X = data.drop(labels=["class"], axis=1)

# 提取目标标签列 'class' 作为 y
# 该列取值为 'p' (有毒 poisonous) 和 'e' (可食用 edible)
y = data["class"]

# 实例化 LabelEncoder 对象，用于将字符串标签转为数值
# LabelEncoder 无必须传入的参数，默认按字母顺序编码
label_encoder = LabelEncoder()

# 调用 fit_transform 方法对目标标签 y 进行编码
# fit 部分：学习标签到数值的映射关系
# transform 部分：根据映射将字符串转为整数
# 返回值：一维 numpy 数组，取值为 0 和 1
y_encoded = label_encoder.fit_transform(y)

# 对所有分类特征进行独热编码（One-Hot Encoding）
# get_dummies 是 pandas 提供的独热编码函数
# 参数 data：待编码的 DataFrame
# 参数 columns：指定需要编码的列，不指定则自动编码所有类别型列
# 独热编码可避免模型错误认为类别间存在大小顺序关系
X_encoded = pd.get_dummies(X)

# ============================================================
# 3. 拆分训练集与测试集
# ============================================================

# 使用 train_test_split 划分训练集和测试集
# 参数 *arrays：待划分的特征矩阵与标签向量，可传入多个
# 参数 test_size：测试集占比，0.2 表示 20% 数据作为测试集
# 参数 random_state：随机种子，设为固定整数保证每次运行划分结果一致
# 参数 stratify：按标签分层抽样，保证训练集和测试集中类别比例与原数据一致
# 返回值：按顺序返回 X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ============================================================
# 4. 构建 XGBoost 模型，手动指定弱分类器
# ============================================================

# 实例化 XGBClassifier 分类器对象，手动指定弱分类器类型
# 参数 booster：弱分类器（基学习器）类型
#   - 'gbtree'：使用 CART 决策树作为弱分类器（默认，最常用）
#   - 'gblinear'：使用线性模型作为弱分类器
#   - 'dart'：带 dropout 的树模型
# 参数 n_estimators：弱分类器的总数量，即迭代轮数
# 参数 max_depth：单棵决策树的最大深度，控制单棵树的复杂度
# 参数 learning_rate：学习率，每棵树的贡献权重，越小越保守
# 参数 objective：目标函数，'binary:logistic' 表示二分类逻辑回归
# 参数 random_state：随机种子，保证结果可复现
# 参数 eval_metric：评估指标，'logloss' 为对数损失
xgb_model = XGBClassifier(
    booster="gbtree",       # 手动指定弱分类器为决策树（CART树）
    n_estimators=100,       # 弱分类器（决策树）的总数量
    max_depth=3,            # 单棵决策树的最大深度
    learning_rate=0.1,      # 学习率，收缩每棵树的贡献
    objective="binary:logistic",  # 二分类目标函数
    random_state=42,        # 随机种子
    eval_metric="logloss"
)

# ============================================================
# 5. 训练模型
# ============================================================

# 调用 fit 方法在训练集上训练模型
# 参数 X：训练集特征矩阵
# 参数 y：训练集标签向量
# 该方法会依次训练每一棵弱分类器，逐步拟合残差
xgb_model.fit(X_train, y_train)

# ============================================================
# 6. 计算并输出训练集与测试集精度
# ============================================================

# 调用 predict 方法对训练集进行预测
# 参数 X：待预测的特征矩阵
# 返回值：预测的类别标签数组（0 或 1）
y_train_pred = xgb_model.predict(X_train)

# 调用 predict 方法对测试集进行预测
y_test_pred = xgb_model.predict(X_test)

# 计算训练集准确率
# 参数 y_true：真实标签
# 参数 y_pred：预测标签
# 返回值：0~1 之间的浮点数，表示准确率
train_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)

# 计算测试集准确率
test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)

# 打印训练集精度，格式化保留 4 位小数
print(f"训练集精度：{train_accuracy:.4f}")

# 打印测试集精度，格式化保留 4 位小数
print(f"测试集精度：{test_accuracy:.4f}")

# ============================================================
# 7. 输入测试数据，打印预测结果
# ============================================================

# 构造一条测试样本（取测试集中第一条样本作为示例）
# iloc[0] 取第 0 行数据，values 转为 numpy 数组
# reshape(1, -1) 将一维数组转为二维，满足 sklearn 输入格式要求
test_sample = X_test.iloc[0].values.reshape(1, -1)

# 调用 predict 方法对单条测试样本进行预测
# 返回值：包含一个元素的数组，取值 0 或 1
pred_label_num = xgb_model.predict(test_sample)

# 使用 LabelEncoder 的 inverse_transform 方法
# 将数值预测结果反向转换回原始字符串标签（'e' 或 'p'）
# 参数 y：数值标签数组
# 返回值：原始字符串标签数组
pred_label_str = label_encoder.inverse_transform(pred_label_num)

# 打印预测结果
# pred_label_num[0] 取出数组中的第一个数值
# pred_label_str[0] 取出数组中的第一个字符串标签
print(f"\n测试数据预测结果（数值编码）：{pred_label_num[0]}")
print(f"测试数据预测结果（原始标签）：{pred_label_str[0]}")
print("标签含义：e = 可食用(edible)，p = 有毒(poisonous)")
