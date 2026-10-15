# ========================== 导入所需库 ==========================

# 导入pandas库，用于读取和处理CSV表格数据
import pandas as pd

# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从sklearn.model_selection模块导入train_test_split函数，用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# 从sklearn.preprocessing模块导入LabelEncoder类，用于将分类标签（字符串）编码为整数
from sklearn.preprocessing import LabelEncoder

# 从sklearn.metrics模块导入accuracy_score函数，用于计算分类准确率
from sklearn.metrics import accuracy_score

# 导入xgboost库中的XGBClassifier类，这是XGBoost提供的sklearn兼容分类器接口
from xgboost import XGBClassifier

# ========================== 1. 加载数据集 ==========================

# 定义数据集文件的完整路径，指向本地存储的secondary_data.csv文件
data_path = r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv"

# 使用pandas的read_csv函数读取CSV文件，返回一个DataFrame对象
# 参数data_path：CSV文件的路径字符串
df = pd.read_csv(data_path,sep=";")

# 打印数据集的前5行，用于快速查看数据结构和内容
# 无参数，默认显示前5行
print("=" * 60)
print("数据集前5行预览：")
print(df.head())

# 打印数据集的形状（行数, 列数），用于了解数据规模
# 无参数，返回元组(样本数, 特征数+标签列)
print(f"\n数据集形状（样本数, 列数）：{df.shape}")

# ========================== 2. 数据预处理 ==========================

# 将DataFrame的第一列（class列）作为标签y，该列值为'e'(可食用)或'p'(有毒)
# df.iloc[:, 0]表示选取所有行的第0列（即第一列class列）
y = df.iloc[:, 0]

# 将DataFrame中除第一列外的所有列作为特征矩阵X
# df.iloc[:, 1:]表示选取所有行、从第1列到最后一列的所有特征列
X = df.iloc[:, 1:]

# 创建LabelEncoder实例，用于将字符串标签（如'e','p'）转换为整数（如0,1）
# 无参数，初始化一个空的编码器对象
label_encoder_y = LabelEncoder()

# 使用fit_transform方法拟合并转换标签列y，将字符串标签编码为整数
# 参数y：原始字符串标签Series，如['e','p','e',...]
# 返回值：编码后的整数数组，如[0,1,0,...]
y_encoded = label_encoder_y.fit_transform(y)

# 打印标签编码的映射关系，显示每个类别对应的整数值
# label_encoder_y.classes_属性存储了编码时学到的类别数组，按字母顺序排列
print(f"\n标签编码映射：{list(label_encoder_y.classes_)} -> [0, 1]")

# 创建LabelEncoder实例，用于将特征列中的字符串值编码为整数
# 因为secondary_data.csv中所有特征都是分类变量（字符串），XGBoost需要数值输入
label_encoder_X = LabelEncoder()

# 使用apply方法对X的每一列（axis=0表示按列操作）分别进行LabelEncoder编码
# 参数lambda col: label_encoder_X.fit_transform(col)：对每列独立拟合并转换
# 返回值：编码后的DataFrame，所有值均为整数
X_encoded = X.apply(lambda col: label_encoder_X.fit_transform(col))

# 打印编码后特征矩阵的前5行，确认编码结果
# 无参数，默认显示前5行
print("\n编码后特征矩阵前5行：")
print(X_encoded.head())

# ========================== 3. 拆分训练集和测试集 ==========================

# 使用train_test_split函数将特征矩阵和标签拆分为训练集和测试集
# 参数X_encoded：编码后的完整特征矩阵（DataFrame或ndarray）
# 参数y_encoded：编码后的完整标签数组
# 参数test_size=0.2：指定测试集占总数据的20%
# 参数random_state=42：设置随机种子为42，确保每次运行拆分结果一致，便于复现
# 返回值：X_train(训练特征), X_test(测试特征), y_train(训练标签), y_test(测试标签)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

# 打印训练集和测试集的样本数量，确认拆分比例是否正确
# X_train.shape[0]获取训练集样本数，X_test.shape[0]获取测试集样本数
print(f"\n训练集样本数：{X_train.shape[0]}")
print(f"测试集样本数：{X_test.shape[0]}")

# ========================== 4. 手动指定弱分类器并构建XGBoost模型 ==========================

# 创建XGBClassifier实例，手动指定弱分类器（基学习器）的类型和参数
# XGBoost中的弱分类器通过以下参数手动指定：
#
# 参数booster='gbtree'：指定弱分类器（基学习器）类型为梯度提升决策树(CART树)
#   可选值：'gbtree'(树模型)、'gblinear'(线性模型)、'dart'(带dropout的树模型)
#   这里选择'gbtree'表示每棵弱分类器都是一棵CART决策树
#
# 参数n_estimators=100：指定弱分类器（决策树）的总数量为100棵
#   即Boosting过程中串行训练100棵决策树，每棵树修正前一棵树的残差
#
# 参数max_depth=6：指定每棵弱分类器（决策树）的最大深度为6层
#   控制单个弱分类器的复杂度，深度越大模型越复杂，越容易过拟合
#
# 参数min_child_weight=3：指定叶子节点上样本的最小权重和为3
#   若某次分裂后某叶子节点的样本权重和小于3，则不进行该分裂
#   用于控制弱分类器的正则化程度，值越大越保守
#
# 参数learning_rate=0.1：指定学习率（步长缩减因子）为0.1
#   每棵弱分类器的预测结果会乘以该系数后再累加到最终预测中
#   较小的学习率需要更多的弱分类器数量，但通常能获得更好的泛化性能
#
# 参数subsample=0.8：指定训练每棵弱分类器时的行采样比例为80%
#   即每棵树只使用80%的训练样本进行训练，引入随机性防止过拟合
#
# 参数colsample_bytree=0.8：指定训练每棵弱分类器时的列采样比例为80%
#   即每棵树只使用80%的特征进行分裂选择，进一步增加随机性
#
# 参数gamma=0.1：指定节点分裂所需的最小损失函数减少量（最小分裂增益）为0.1
#   只有当分裂带来的增益大于gamma时才执行分裂，用于控制弱分类器的剪枝
#
# 参数reg_alpha=0.1：指定L1正则化系数为0.1
#   对叶子节点权重施加L1惩罚，促进稀疏性，防止弱分类器过拟合
#
# 参数reg_lambda=1.0：指定L2正则化系数为1.0
#   对叶子节点权重施加L2惩罚，平滑权重值，防止弱分类器过拟合
#
# 参数objective='binary:logistic'：指定学习目标函数为二分类逻辑回归
#   因为蘑菇数据集是二分类问题（可食用/有毒），使用logistic输出概率
#
# 参数eval_metric='logloss'：指定评估指标为对数损失（交叉熵损失）
#   用于在训练过程中监控模型性能
#
# 参数random_state=42：设置随机种子为42，确保模型训练结果可复现
#
# 参数use_label_encoder=False：禁用内部标签编码器（新版xgboost默认不需要）
#   因为我们已经手动使用LabelEncoder对标签进行了编码
#
# 参数verbosity=1：设置训练过程的日志输出级别为1，输出基本训练信息
#   0=静默，1=警告，2=信息，3=调试
xgb_model = XGBClassifier(
    booster='gbtree',
    n_estimators=100,
    max_depth=3,
    min_child_weight=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    verbosity=1
)

# 打印模型配置信息，确认弱分类器参数设置
# 无参数，返回模型所有参数的字符串表示
print("\n" + "=" * 60)
print("XGBoost模型弱分类器配置：")
print(f"  弱分类器类型(booster)：gbtree（CART决策树）")
print(f"  弱分类器数量(n_estimators)：100棵")
print(f"  单棵弱分类器最大深度(max_depth)：6层")
print(f"  叶子节点最小权重和(min_child_weight)：3")
print(f"  学习率(learning_rate)：0.1")
print(f"  行采样比例(subsample)：0.8")
print(f"  列采样比例(colsample_bytree)：0.8")
print(f"  最小分裂增益(gamma)：0.1")
print(f"  L1正则化(reg_alpha)：0.1")
print(f"  L2正则化(reg_lambda)：1.0")

# ========================== 5. 训练模型 ==========================

# 调用fit方法在训练集上训练XGBoost模型
# 参数X_train：训练集特征矩阵，形状为(训练样本数, 特征数)
# 参数y_train：训练集标签数组，形状为(训练样本数,)，值为0或1
# 该方法会串行训练100棵决策树（弱分类器），每棵树拟合前面所有树的残差
xgb_model.fit(X_train, y_train)

# 打印训练完成提示信息
print("\n模型训练完成！")

# ========================== 6. 输出训练集精度和测试集精度 ==========================

# 使用训练好的模型对训练集进行预测，返回预测标签数组
# 参数X_train：训练集特征矩阵
# 返回值：预测标签数组，形状为(训练样本数,)，值为0或1
y_train_pred = xgb_model.predict(X_train)

# 使用训练好的模型对测试集进行预测，返回预测标签数组
# 参数X_test：测试集特征矩阵
# 返回值：预测标签数组，形状为(测试样本数,)，值为0或1
y_test_pred = xgb_model.predict(X_test)

# 使用accuracy_score函数计算训练集上的分类准确率
# 参数y_train：训练集真实标签数组
# 参数y_train_pred：训练集预测标签数组
# 返回值：准确率浮点数，范围[0, 1]，表示预测正确的样本比例
train_accuracy = accuracy_score(y_train, y_train_pred)

# 使用accuracy_score函数计算测试集上的分类准确率
# 参数y_test：测试集真实标签数组
# 参数y_test_pred：测试集预测标签数组
# 返回值：准确率浮点数，范围[0, 1]，表示预测正确的样本比例
test_accuracy = accuracy_score(y_test, y_test_pred)

# 打印训练集精度，格式化为4位小数
# f-string中.4f表示保留4位小数
print("\n" + "=" * 60)
print(f"训练集精度（Accuracy）：{train_accuracy:.4f}（即 {train_accuracy * 100:.2f}%）")

# 打印测试集精度，格式化为4位小数
print(f"测试集精度（Accuracy）：{test_accuracy:.4f}（即 {test_accuracy * 100:.2f}%）")

# ========================== 7. 输入测试数据并打印预测结果 ==========================

# 从测试集中取出前5条样本作为演示用的测试数据
# X_test.iloc[:5]表示取测试集DataFrame的前5行所有列
sample_test_data = X_test.iloc[:5]

# 取出对应的真实标签，用于与预测结果对比
# y_test[:5]表示取测试集标签数组的前5个元素
sample_true_labels = y_test[:5]

# 使用训练好的模型对这5条测试数据进行预测
# 参数sample_test_data：待预测的特征数据，形状为(5, 特征数)
# 返回值：预测标签数组，形状为(5,)，值为0或1
sample_predictions = xgb_model.predict(sample_test_data)

# 使用predict_proba方法获取预测概率，返回每个样本属于各类别的概率
# 参数sample_test_data：待预测的特征数据
# 返回值：概率矩阵，形状为(5, 2)，第0列为属于类别0的概率，第1列为属于类别1的概率
sample_probabilities = xgb_model.predict_proba(sample_test_data)

# 打印分隔线和标题
print("\n" + "=" * 60)
print("测试数据预测结果（前5条样本）：")
print("=" * 60)

# 使用for循环逐条打印每个测试样本的预测详情
# enumerate(sample_predictions)返回(索引, 预测值)的迭代器
for i, pred in enumerate(sample_predictions):
    # 将整数预测标签转换回原始字符串标签（'e'或'p'）
    # label_encoder_y.inverse_transform()方法将整数编码还原为原始字符串
    # 参数[pred]：包含单个整数的列表
    # 返回值：包含原始字符串的数组，取[0]得到字符串
    pred_label = label_encoder_y.inverse_transform([pred])[0]

    # 将真实整数标签转换回原始字符串标签
    # 参数[sample_true_labels[i]]：包含单个真实标签整数的列表
    true_label = label_encoder_y.inverse_transform([sample_true_labels[i]])[0]

    # 获取该样本属于"有毒(p)"类别的预测概率
    # sample_probabilities[i][1]表示第i个样本属于类别1（有毒）的概率
    poison_prob = sample_probabilities[i][1]

    # 获取该样本属于"可食用(e)"类别的预测概率
    # sample_probabilities[i][0]表示第i个样本属于类别0（可食用）的概率
    edible_prob = sample_probabilities[i][0]

    # 判断预测是否正确，正确显示"✓"，错误显示"✗"
    # 三元表达式：如果预测值等于真实值则为"✓"，否则为"✗"
    correct_mark = "✓" if pred == sample_true_labels[i] else "✗"

    # 打印该样本的完整预测信息
    # 包含：样本序号、真实标签、预测标签、可食用概率、有毒概率、是否正确
    print(f"  样本 {i + 1}：真实标签 = {true_label}，"
          f"预测标签 = {pred_label}，"
          f"可食用概率 = {edible_prob:.4f}，"
          f"有毒概率 = {poison_prob:.4f}  {correct_mark}")

# 打印最终总结信息
print("\n" + "=" * 60)
print("说明：标签 'e' = edible（可食用），'p' = poisonous（有毒）")
print("=" * 60)