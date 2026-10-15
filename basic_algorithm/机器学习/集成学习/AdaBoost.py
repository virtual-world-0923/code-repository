# ============================================================================
# AdaBoost算法练习 —— 蘑菇数据集（Mushroom Dataset）
# 数据集来源：UCI Secondary Mushroom Dataset
# 功能：加载数据、拆分数据集、手动指定弱分类器、训练AdaBoost模型、评估精度、预测新数据
# 适用版本：scikit-learn >= 1.6（algorithm参数已移除，默认SAMME）
# ============================================================================

# 导入pandas库，用于读取CSV文件和处理表格数据
import pandas as pd
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 从sklearn.model_selection模块导入train_test_split函数，用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split
# 从sklearn.preprocessing模块导入LabelEncoder类，用于将分类标签（字符串）编码为整数
from sklearn.preprocessing import LabelEncoder
# 从sklearn.ensemble模块导入AdaBoostClassifier类，用于构建AdaBoost集成分类器
from sklearn.ensemble import AdaBoostClassifier
# 从sklearn.tree模块导入DecisionTreeClassifier类，用作AdaBoost的弱分类器（基学习器）
from sklearn.tree import DecisionTreeClassifier
# 从sklearn.metrics模块导入accuracy_score函数，用于计算分类准确率
from sklearn.metrics import accuracy_score
# 导入sklearn库，用于打印版本号确认当前环境版本
import sklearn

# ============================================================================
# 第一步：打印环境版本信息（便于排查兼容性问题）
# ============================================================================

# 打印当前scikit-learn的版本号，确认API参数兼容性
print(f"scikit-learn 版本: {sklearn.__version__}")  # __version__属性返回版本号字符串
# 打印当前numpy的版本号
print(f"numpy 版本: {np.__version__}")  # np.__version__返回numpy版本号
# 打印当前pandas的版本号
print(f"pandas 版本: {pd.__version__}")  # pd.__version__返回pandas版本号

# ============================================================================
# 第二步：加载数据集
# ============================================================================

# 定义数据集文件的完整路径，指向本地存储的secondary_data.csv文件
data_path = r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv"

# 使用pandas的read_csv函数读取CSV文件
# 参数说明：
#   data_path: 文件路径，指向本地CSV文件
#   sep=';': 【关键】指定分隔符为分号（UCI蘑菇数据集使用分号;而非逗号,作为列分隔符）
#   na_values='?': 【关键】将数据中的'?'符号识别为缺失值NaN（该数据集用?表示缺失）
df = pd.read_csv(data_path, sep=';', na_values='?')

# 打印数据集的前5行，用于快速查看数据结构和内容
print("\n" + "=" * 60)  # 打印换行+60个等号作为分隔线
print("数据集前5行预览：")  # 打印提示信息
print(df.head())  # 调用head()方法显示前5行数据，默认参数n=5

# 打印数据集的形状（行数, 列数），用于了解数据规模
print(f"\n数据集形状（样本数, 特征数+标签列）: {df.shape}")  # df.shape返回(行数, 列数)的元组
# 打印各列的数据类型，确认读取是否正确
print(f"\n各列数据类型:\n{df.dtypes}")  # df.dtypes返回每列数据类型的Series对象
# 打印缺失值统计，查看每列有多少缺失值
print(f"\n各列缺失值数量:\n{df.isnull().sum()}")  # isnull()标记缺失值为True，sum()按列求和统计

# ============================================================================
# 第三步：数据预处理 —— 分离特征和标签，处理缺失值，进行编码
# ============================================================================

# 提取标签列（第一列'class'），该列值为'e'(可食用)或'p'(有毒)，作为目标变量y
y = df.iloc[:, 0]  # iloc[:, 0]表示选取所有行的第0列（即第一列class列）
# 提取特征列（除第一列外的所有列），作为输入特征矩阵X
X = df.iloc[:, 1:]  # iloc[:, 1:]表示选取所有行的第1列到最后一列（即所有特征列）

# 打印标签列的唯一值，确认分类类别（应为'e'和'p'）
print(f"\n标签类别: {y.unique()}")  # unique()返回数组中所有不重复的值
# 打印特征数量
print(f"特征数量: {X.shape[1]}")  # X.shape[1]获取特征列数
# 打印特征列的列名列表，查看所有特征名称
print(f"特征列名: {list(X.columns)}")  # X.columns返回所有列名，list()转为列表

# ---------- 处理缺失值 ----------
# 对特征矩阵X中的缺失值进行填充，使用每列的众数（出现频率最高的值）填充
# 参数说明：
#   X.mode(): 计算每列的众数，返回一个DataFrame（可能有多行众数）
#   .iloc[0]: 取众数结果的第一行（如果某列有多个众数，取第一个）
#   fillna(): 用传入的Series值填充对应列的NaN
X_filled = X.fillna(X.mode().iloc[0])  # 用各列众数填充该列的缺失值

# 打印填充后剩余缺失值总数，确认所有缺失值已被处理（应为0）
print(f"\n填充后剩余缺失值总数: {X_filled.isnull().sum().sum()}")  # 两次sum()得到全局缺失值总数

# ---------- 标签编码 ----------
# 创建LabelEncoder实例，用于将字符串标签编码为整数（如'e'->0, 'p'->1）
label_encoder_y = LabelEncoder()  # 实例化LabelEncoder对象，无需传入参数
# 对标签y进行拟合和转换，将字符串标签转为整数编码，返回编码后的numpy数组
y_encoded = label_encoder_y.fit_transform(y)  # fit_transform(y)：先学习映射关系再转换，参数y为原始标签Series
# 打印编码后的标签类别映射关系，展示原始值与编码值的对应关系
print(f"标签编码映射: {dict(zip(label_encoder_y.classes_, range(len(label_encoder_y.classes_))))}")  # classes_存储原始类别列表

# 【关键】对特征矩阵X_filled的每一列分别创建独立的LabelEncoder进行编码
# 使用字典推导式，为每一列创建独立的LabelEncoder（避免不同列共享编码器导致映射混乱）
# 参数说明：
#   col: 当前正在处理的列名（字符串）
#   X_filled[col]: 取出该列的Series数据
#   .astype(str): 将列数据强制转为字符串类型，确保LabelEncoder能处理所有值
#   LabelEncoder().fit_transform(): 对该列独立进行拟合和编码转换
# 最终将所有编码后的列合并为一个新的DataFrame
X_encoded_df = pd.DataFrame({
    col: LabelEncoder().fit_transform(X_filled[col].astype(str))  # 对每列独立创建编码器并编码
    for col in X_filled.columns  # 遍历所有特征列名
})  # 字典推导式构建DataFrame，键为列名，值为编码后的整数数组

# 将编码后的DataFrame转换为numpy二维数组，便于后续sklearn模型处理
X_encoded = X_encoded_df.to_numpy()  # .values属性将DataFrame转为二维numpy数组，形状为(样本数, 特征数)

# 打印编码后特征矩阵的形状，确认数据维度正确（应为61069行 × 特征数列）
print(f"编码后特征矩阵形状: {X_encoded.shape}")  # 输出(样本数, 特征数)
# 打印编码后标签数组的形状，确认标签数量与样本数一致（应为61069）
print(f"编码后标签数组形状: {y_encoded.shape}")  # 输出(样本数,)

# ============================================================================
# 第四步：拆分训练集和测试集
# ============================================================================

# 使用train_test_split函数将数据集拆分为训练集和测试集
# 参数说明：
#   X_encoded: 编码后的特征矩阵，形状为(61069, 特征数)，作为输入数据
#   y_encoded: 编码后的标签数组，形状为(61069,)，作为目标变量
#   test_size=0.3: 测试集占总数据的30%，训练集占70%
#   random_state=42: 随机种子，设为42保证每次运行拆分结果一致，便于复现
#   stratify=y_encoded: 分层抽样，确保训练集和测试集中各类别比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,           # 第一个参数：编码后的完整特征矩阵，形状(61069, 特征数)
    y_encoded,           # 第二个参数：编码后的完整标签数组，形状(61069,)
    test_size=0.3,       # 测试集比例参数：30%的数据作为测试集
    random_state=42,     # 随机种子参数：固定为42确保可复现
    stratify=y_encoded   # 分层抽样参数：按标签比例分层，保持类别平衡
)

# 打印训练集的样本数量，确认拆分结果
print(f"\n训练集样本数: {X_train.shape[0]}")  # X_train.shape[0]获取训练集行数（样本数）
# 打印测试集的样本数量，确认拆分结果
print(f"测试集样本数: {X_test.shape[0]}")  # X_test.shape[0]获取测试集行数（样本数）
# 打印训练集中各类别的样本数量分布，验证分层抽样效果
print(f"训练集标签分布: {np.bincount(y_train)}")  # np.bincount统计每个整数标签出现的次数(即每个类别出现的次数)
# 打印测试集中各类别的样本数量分布，验证分层抽样效果
print(f"测试集标签分布: {np.bincount(y_test)}")  # np.bincount统计每个整数标签出现的次数(即每个类别出现的次数)

# ============================================================================
# 第五步：手动指定弱分类器（基学习器）
# ============================================================================

# 手动创建决策树弱分类器实例，作为AdaBoost的基学习器
# 参数说明：
#   max_depth=1: 决策树最大深度为1（即决策树桩/Decision Stump），这是AdaBoost最常用的弱分类器
#                深度为1意味着树只有一个根节点和两个叶节点，是最简单的分类器
#   min_samples_split=2: 内部节点再划分所需的最小样本数为2，默认值，防止过拟合
#   min_samples_leaf=1: 叶节点最少样本数为1，默认值，允许叶节点只有一个样本
#   random_state=42: 随机种子，保证决策树构建过程可复现
weak_classifier = DecisionTreeClassifier(
    max_depth=1,             # 最大深度参数：设为1，构建决策树桩（最弱的分类器）
    min_samples_split=2,     # 最小分裂样本数参数：节点至少需要2个样本才继续分裂
    min_samples_leaf=1,      # 叶节点最小样本数参数：每个叶节点至少包含1个样本
    random_state=42          # 随机种子参数：固定为42确保结果可复现
)

# 打印弱分类器的配置信息，确认参数设置正确
print(f"\n弱分类器类型: {type(weak_classifier).__name__}")  # type()获取对象类型，__name__获取类名字符串
# 打印弱分类器的具体参数，展示手动指定的所有参数值
print(f"弱分类器参数: max_depth={weak_classifier.max_depth}, "
      f"min_samples_split={weak_classifier.min_samples_split}, "
      f"min_samples_leaf={weak_classifier.min_samples_leaf}")  # 访问对象属性打印各参数值

# ============================================================================
# 第六步：构建AdaBoost集成分类器
# ============================================================================

# 创建AdaBoost分类器实例
# 参数说明：
#   estimator=weak_classifier: 基学习器（弱分类器），传入上面手动创建的决策树桩
#   n_estimators=50: 弱分类器的数量（即迭代轮数），设为50表示训练50个决策树桩
#                    数量越多模型越复杂，但过多可能导致过拟合
#   learning_rate=1.0: 学习率，控制每个弱分类器对最终结果的贡献权重
#                      设为1.0表示不缩放，每个分类器按原始权重贡献
#                      较小的学习率需要更多的弱分类器但通常能获得更好的泛化性能
#   random_state=42: 随机种子，保证AdaBoost训练过程可复现
ada_boost_classifier = AdaBoostClassifier(
    estimator=weak_classifier,   # 基学习器参数：传入手动指定的决策树桩弱分类器
    n_estimators=50,             # 弱分类器数量参数：集成中包含50个弱分类器
    learning_rate=1.0,           # 学习率参数：每个弱分类器的权重缩放因子为1.0
    random_state=42              # 随机种子参数：固定为42确保结果可复现
)  # 【注意】不再传入algorithm参数，sklearn>=1.6已移除该参数，默认使用SAMME算法

# 打印AdaBoost分类器的配置信息，确认所有参数设置
print(f"\nAdaBoost分类器配置:")  # 打印标题信息
print(f"  弱分类器: DecisionTreeClassifier(max_depth=1)")  # 打印弱分类器类型信息
print(f"  弱分类器数量(n_estimators): {ada_boost_classifier.n_estimators}")  # 打印弱分类器数量
print(f"  学习率(learning_rate): {ada_boost_classifier.learning_rate}")  # 打印学习率值

# ============================================================================
# 第七步：训练AdaBoost模型
# ============================================================================

# 使用训练集数据拟合（训练）AdaBoost分类器模型
# 参数说明：
#   X_train: 训练集特征矩阵，形状为(训练样本数, 特征数)
#   y_train: 训练集标签数组，形状为(训练样本数,)
# 该方法会迭代训练n_estimators个弱分类器，每轮根据上一轮的错误调整样本权重
ada_boost_classifier.fit(X_train, y_train)  # fit方法执行模型训练，X_train为特征，y_train为标签

# 打印训练完成提示信息，显示实际训练的弱分类器数量
print(f"\n模型训练完成！共训练了 {len(ada_boost_classifier.estimators_)} 个弱分类器")  # estimators_属性存储所有已训练的弱分类器列表

# ============================================================================
# 第八步：输出训练集精度和测试集精度
# ============================================================================

# 使用训练好的模型对训练集进行预测，得到训练集的预测标签
# 参数X_train：训练集特征矩阵，模型对每个样本输出预测类别
y_train_pred = ada_boost_classifier.predict(X_train)  # predict方法返回预测标签数组，形状与y_train相同

# 使用训练好的模型对测试集进行预测，得到测试集的预测标签
# 参数X_test：测试集特征矩阵，模型对每个样本输出预测类别
y_test_pred = ada_boost_classifier.predict(X_test)  # predict方法返回预测标签数组，形状与y_test相同

# 计算训练集的分类准确率
# 参数说明：
#   y_train: 训练集的真实标签数组（第一个参数）
#   y_train_pred: 模型对训练集的预测标签数组（第二个参数）
# 返回值为0到1之间的浮点数，表示正确预测的比例
train_accuracy = accuracy_score(y_train, y_train_pred)  # accuracy_score(真实标签, 预测标签)计算准确率

# 计算测试集的分类准确率
# 参数说明：
#   y_test: 测试集的真实标签数组（第一个参数）
#   y_test_pred: 模型对测试集的预测标签数组（第二个参数）
# 返回值为0到1之间的浮点数，表示正确预测的比例
test_accuracy = accuracy_score(y_test, y_test_pred)  # accuracy_score(真实标签, 预测标签)计算准确率

# 打印训练集精度，格式化为百分比显示，保留4位小数
print(f"\n{'=' * 60}")  # 打印60个等号作为分隔线
print(f"训练集精度 (Train Accuracy): {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")  # :.4f保留4位小数
# 打印测试集精度，格式化为百分比显示，保留4位小数
print(f"测试集精度 (Test Accuracy):  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")  # :.4f保留4位小数
print(f"{'=' * 60}")  # 打印60个等号作为分隔线

# ============================================================================
# 第九步：输入测试数据，打印预测结果
# ============================================================================

# 打印提示信息，告知用户将输入自定义测试数据进行预测
print(f"\n{'=' * 60}")  # 打印60个等号作为分隔线
print("输入测试数据进行预测：")  # 打印提示信息
print(f"{'=' * 60}")  # 打印60个等号作为分隔线

# 设置numpy随机种子为0，保证随机选取结果可复现
np.random.seed(0)  # 设置全局随机种子为0，确保每次运行选取相同样本
# 从测试集中随机选取5个样本的索引，用于演示预测
# 参数说明：
#   X_test.shape[0]: 测试集总样本数，作为索引范围上限（从0到该值-1中选取）
#   size=5: 选取5个样本
#   replace=False: 不放回抽样，确保选取的样本不重复
test_indices = np.random.choice(X_test.shape[0], size=5, replace=False)  # 返回5个不重复的随机索引数组

# 根据随机选取的索引，从测试集中提取对应的特征数据作为输入测试数据
# 参数test_indices：之前随机生成的5个样本索引数组
test_data = X_test[test_indices]  # 使用索引数组从测试集中提取对应行，形状(5, 特征数)

# 根据随机选取的索引，从测试集标签中提取对应的真实标签（用于对比验证）
# 参数test_indices：之前随机生成的5个样本索引数组
test_true_labels = y_test[test_indices]  # 使用索引数组从标签数组中提取对应值，形状(5,)

# 使用训练好的AdaBoost模型对输入的测试数据进行预测
# 参数test_data：形状为(5, 特征数)的测试数据矩阵，包含5个待预测样本
test_predictions = ada_boost_classifier.predict(test_data)  # predict返回每个样本的预测类别（整数编码），形状(5,)

# 获取每个预测样本属于各类别的概率，用于展示模型的置信度
# 参数test_data：形状为(5, 特征数)的测试数据矩阵
test_probabilities = ada_boost_classifier.predict_proba(test_data)  # predict_proba返回形状(5, 2)的概率数组

# 遍历每个测试样本，逐一打印预测结果和详细信息
# 参数range(len(test_data))：生成0到4的索引序列，共5个样本
for i in range(len(test_data)):  # i为当前样本的索引，从0到4
    # 打印当前样本的序号（从1开始计数，更直观）
    print(f"\n--- 测试样本 {i + 1} ---")  # i+1将索引从0起始转为1起始显示
    # 打印该样本的特征向量（编码后的整数值），展示输入数据
    print(f"  输入特征（编码后）: {test_data[i]}")  # test_data[i]为第i个样本的特征数组
    # 打印该样本的真实标签（整数编码），用于与预测结果对比
    print(f"  真实标签（编码）: {test_true_labels[i]}")  # test_true_labels[i]为第i个样本的真实标签整数
    # 将真实标签的整数编码转换回原始字符串标签（如0->'e', 1->'p'）
    true_label_str = label_encoder_y.inverse_transform([test_true_labels[i]])[0]  # inverse_transform将整数转回原始标签，[0]取第一个元素
    # 打印该样本的预测标签（整数编码）
    print(f"  预测标签（编码）: {test_predictions[i]}")  # test_predictions[i]为第i个样本的预测标签整数
    # 将预测标签的整数编码转换回原始字符串标签（如0->'e'表示可食用, 1->'p'表示有毒）
    pred_label_str = label_encoder_y.inverse_transform([test_predictions[i]])[0]  # inverse_transform将整数转回原始标签，[0]取第一个元素
    # 打印原始字符串形式的真实标签及其中文含义
    print(f"  真实类别: '{true_label_str}' ({'可食用' if true_label_str == 'e' else '有毒'})")  # 三元表达式根据标签值显示中文
    # 打印原始字符串形式的预测标签及其中文含义
    print(f"  预测类别: '{pred_label_str}' ({'可食用' if pred_label_str == 'e' else '有毒'})")  # 三元表达式根据标签值显示中文
    # 打印模型对该样本预测为各类别的概率，展示预测置信度
    print(f"  预测概率: 可食用(e)={test_probabilities[i][0]:.4f}, 有毒(p)={test_probabilities[i][1]:.4f}")  # 概率保留4位小数
    # 判断预测是否正确，生成对比结果字符串
    is_correct = "✓ 正确" if test_predictions[i] == test_true_labels[i] else "✗ 错误"  # 三元表达式判断预测值是否等于真实值
    # 打印预测正确或错误的标记
    print(f"  预测结果: {is_correct}")  # 输出判断结果字符串

# 打印程序结束提示信息
print(f"\n{'=' * 60}")  # 打印60个等号作为结束分隔线
print("AdaBoost蘑菇分类练习程序运行结束！")  # 打印结束信息
print(f"{'=' * 60}")  # 打印60个等号作为结束分隔线