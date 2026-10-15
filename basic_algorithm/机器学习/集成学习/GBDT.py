# ============================================================
# GBDT（梯度提升决策树）算法练习 —— 蘑菇数据集分类（pandas 3.x 兼容版）
# sklearn 版本：1.9.0
# pandas 版本：3.x（兼容 Copy-on-Write 和 str dtype）
# 修正内容：
#   1. select_dtypes 同时包含 'object' 和 'str'，兼容 pandas 2 和 3
#   2. 使用 X[col] = X[col].fillna(...) 替代 inplace=True，避免 CoW 报错
# ============================================================

# 导入 pandas 库，用于读取 CSV 文件和处理表格数据
import pandas as pd

# 导入 numpy 库，用于数值计算和数组操作
import numpy as np

# 从 sklearn.preprocessing 模块导入 LabelEncoder 类
# 作用：将分类变量（字符串标签）转换为整数编码
from sklearn.preprocessing import LabelEncoder

# 从 sklearn.model_selection 模块导入 train_test_split 函数
# 作用：将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# 从 sklearn.ensemble 模块导入 GradientBoostingClassifier 类
# 作用：创建 GBDT（梯度提升决策树）分类器模型
from sklearn.ensemble import GradientBoostingClassifier

# 从 sklearn.metrics 模块导入 accuracy_score 函数
# 作用：计算分类准确率（正确预测数 / 总样本数）
from sklearn.metrics import accuracy_score

# ============================================================
# 第一步：加载数据集
# ============================================================

# 定义数据集文件的完整路径
# 参数说明：字符串路径，指向本地 CSV 文件
data_path = r"D:\Repositories\basic_algorithm\database\MushroomDataset\secondary_data.csv"

# 使用 pandas 的 read_csv 函数读取 CSV 文件，返回一个 DataFrame 对象
# 参数说明：data_path 为文件路径；该数据集包含 61068 条样本、20 个特征和 1 个目标列
df = pd.read_csv(data_path,sep=";")

# 打印数据集的前 5 行，用于初步查看数据结构和内容
# 参数说明：无参数，默认显示前 5 行
print("=" * 60)  # 打印分隔线，宽度为 60 个等号字符
print("数据集前 5 行预览：")  # 打印提示信息
print(df.head())  # 调用 head() 方法显示 DataFrame 的前 5 行数据

# 打印数据集的形状（行数, 列数），用于了解数据规模
# 参数说明：shape 是 DataFrame 的属性，返回元组 (n_rows, n_columns)
print(f"\n数据集形状（样本数, 列数）：{df.shape}")  # 格式化字符串输出形状信息

# 打印每列的数据类型，用于区分数值型列和分类型列
# 参数说明：dtypes 属性返回每列的数据类型（float64 / object / str 等）
print("\n各列数据类型：")  # 打印提示信息
print(df.dtypes)  # 显示所有列的数据类型

# 打印每列的缺失值数量，用于了解数据质量
# 参数说明：isnull() 返回布尔矩阵，sum() 按列求和得到每列 NaN 数量
print("\n各列缺失值数量：")  # 打印提示信息
print(df.isnull().sum())  # 显示每列的缺失值统计

# ============================================================
# 第二步：分离特征和标签
# ============================================================

# 提取目标变量（标签列），列名为 'class'
# 参数说明：df['class'] 返回 'class' 列的 Series 对象，值为 'e'（可食用）或 'p'（有毒）
y = df['class']

# 提取特征矩阵，即删除 'class' 列后的所有剩余列
# 参数说明：axis=1 表示按列方向删除；返回一个新的 DataFrame，包含 20 个特征列
X = df.drop('class', axis=1)

# ============================================================
# 第三步：处理缺失值（关键修正！）
# ============================================================
# 说明：secondary_data.csv 中多个列存在 NaN 缺失值
#       数值型列用中位数填充（不受极端值影响）
#       分类型列用众数填充（最常见的类别）

# 获取所有数值型列的列名列表（dtype 为 float64 或 int64 的列）
# 参数说明：select_dtypes(include=['number']) 筛选数值型列；.columns 获取列名
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

# 获取所有分类型列的列名列表
# 【修正】pandas 3.x 中字符串列的 dtype 为 'str' 而非 'object'
#         为兼容 pandas 2.x 和 3.x，同时传入 'object' 和 'str'
# 参数说明：include=['object', 'str'] 同时匹配旧版 object 类型和新版 str 类型
categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()

# 打印数值型列和分类型列的名称，确认分类结果
# 参数说明：直接打印列表内容
print(f"\n数值型列（共 {len(numeric_cols)} 个）：{numeric_cols}")  # 打印数值型列名
print(f"分类型列（共 {len(categorical_cols)} 个）：{categorical_cols}")  # 打印分类型列名

# 对数值型列使用中位数填充缺失值
# 【修正】pandas 3.x 启用 Copy-on-Write，不能使用 X[col].fillna(..., inplace=True)
#         必须使用 X[col] = X[col].fillna(...) 的赋值方式
# 参数说明：
#   X[col].median()：计算该列的中位数（第50百分位数），对异常值鲁棒
#   X[col].fillna(value)：返回填充后的新 Series（不修改原对象）
#   X[col] = ...：将填充后的结果赋值回原 DataFrame 的对应列
for col in numeric_cols:  # 遍历每个数值型列名
    # 用该列的中位数填充该列中的所有 NaN 值，并赋值回原列
    # 参数说明：X[col].median() 返回中位数标量；fillna() 返回填充后的新 Series
    X[col] = X[col].fillna(X[col].median())  # 赋值式填充（兼容 CoW）

# 对分类型列使用众数（出现频率最高的值）填充缺失值
# 【修正】同上，使用赋值式替代 inplace=True
# 参数说明：
#   X[col].mode()[0]：返回该列出现次数最多的值（mode() 返回 Series，取第一个元素）
#   X[col].fillna(value)：返回填充后的新 Series
#   X[col] = ...：将填充后的结果赋值回原 DataFrame 的对应列
for col in categorical_cols:  # 遍历每个分类型列名
    # 用该列的众数填充该列中的所有 NaN 值，并赋值回原列
    # 参数说明：X[col].mode()[0] 返回众数（字符串）；fillna() 返回填充后的新 Series
    X[col] = X[col].fillna(X[col].mode()[0])  # 赋值式填充（兼容 CoW）

# 验证缺失值是否已全部处理完毕
# 参数说明：isnull().sum().sum() 返回整个 DataFrame 中 NaN 的总数
print(f"\n处理后剩余缺失值总数：{X.isnull().sum().sum()}")  # 应输出 0

# ============================================================
# 第四步：对分类变量进行标签编码（仅对分类型列！）
# ============================================================
# 关键修正：只对分类型列进行 LabelEncoder 编码
#           数值型列（cap-diameter, stem-height, stem-width）保持原样不动
#           因为它们已经是数值型，且保留了连续值的大小和距离信息

# 对目标变量 y 进行编码，将 'e'/'p' 转换为 0/1
# 参数说明：LabelEncoder() 创建编码器实例；fit_transform(y) 拟合并转换
# 编码结果：'e'（edible，可食用）→ 0，'p'（poisonous，有毒）→ 1
label_encoder_y = LabelEncoder()  # 创建目标变量编码器实例
y_encoded = label_encoder_y.fit_transform(y)  # 对目标变量进行编码

# 打印目标变量的编码映射关系
# 参数说明：classes_ 属性返回编码后的类别数组，按字母顺序排列
print(f"目标变量编码映射：{list(label_encoder_y.classes_)} → [0, 1]")  # 打印映射

# 仅对分类型特征列进行 LabelEncoder 编码
# 参数说明：遍历 categorical_cols 列表中的每个列名
for col in categorical_cols:  # 遍历每个分类型列名（共 17 个）
    # 为当前分类型列创建一个独立的 LabelEncoder 实例
    # 参数说明：每个列需要独立的编码器，因为各列的类别值不同
    le = LabelEncoder()  # 创建当前列的编码器实例
    # 对当前列进行拟合并转换，将字符串值转换为整数编码
    # 参数说明：fit_transform(X[col]) 学习该列唯一值映射并返回编码后的数组
    # 使用赋值式 X[col] = ... 将编码结果写回原 DataFrame
    X[col] = le.fit_transform(X[col])  # 编码后赋值回原列

# 打印编码后的特征矩阵前 5 行和数据类型，确认编码结果
# 参数说明：head() 显示前 5 行；dtypes 显示各列类型
print("\n编码后的特征矩阵前 5 行：")  # 打印提示信息
print(X.head())  # 显示编码后的数据
print("\n编码后各列数据类型：")  # 打印提示信息
print(X.dtypes)  # 确认所有列现在都是数值型

# ============================================================
# 第五步：拆分训练集和测试集
# ============================================================

# 使用 train_test_split 函数将数据拆分为训练集和测试集
# 参数说明：
#   X：特征矩阵，形状为 (61068, 20)，所有列现在都是数值型
#   y_encoded：编码后的目标变量数组，形状为 (61068,)
#   test_size=0.2：测试集占总数据的 20%（即约 12214 条样本）
#   random_state=42：随机种子，设为固定值 42 以保证每次运行拆分结果一致
#   stratify=y_encoded：分层抽样，确保训练集和测试集中各类别比例与原始数据一致
# 返回值：X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # 特征矩阵（自变量）
    y_encoded,          # 目标变量（因变量）
    test_size=0.2,      # 测试集比例为 20%
    random_state=42,    # 随机种子，保证结果可复现
    stratify=y_encoded  # 分层抽样，保持类别比例一致
)

# 打印训练集和测试集的样本数量，确认拆分结果
# 参数说明：shape[0] 获取数组的第一维大小，即样本数量
print(f"\n训练集样本数：{X_train.shape[0]}")  # 打印训练集样本数（约 48854）
print(f"测试集样本数：{X_test.shape[0]}")    # 打印测试集样本数（约 12214）

# ============================================================
# 第六步：手动指定弱分类器参数，构建 GBDT 模型
# ============================================================
# 说明：GBDT 的弱分类器是 CART 决策树
#       通过以下参数手动指定弱分类器的结构和行为

# 创建 GradientBoostingClassifier 实例，手动指定所有关键参数
# 参数说明：
#   loss='log_loss'：损失函数，对数损失（负对数似然），适用于分类任务
#   learning_rate=0.1：学习率（步长），控制每棵树对最终结果的贡献权重
#   n_estimators=200：弱分类器（决策树）的数量，即迭代轮数
#   max_depth=5：弱分类器（决策树）的最大深度，控制树的复杂度
#   min_samples_split=10：内部节点再划分所需的最小样本数，防止过拟合
#   min_samples_leaf=5：每个叶子节点必须包含的最小样本数，防止过拟合
#   max_leaf_nodes=31：最大叶子节点数，限制树的规模（2^5-1=31）
#   min_impurity_decrease=0.01：最小不纯度减少阈值，用于剪枝
#   max_features='sqrt'：每个节点分裂时考虑的最大特征数（√20≈4）
#   subsample=0.8：子采样比例，每轮随机抽取 80% 样本训练当前树
#   ccp_alpha=0.0：代价复杂度剪枝参数，0.0 表示不剪枝
#   random_state=42：随机种子，保证结果可复现
#   verbose=0：训练日志级别，0 表示静默
gbdt_model = GradientBoostingClassifier(
    loss='log_loss',              # 损失函数：对数损失
    learning_rate=0.1,            # 学习率：0.1
    n_estimators=100,             # 弱分类器数量：200 棵决策树
    max_depth=3,                  # 弱分类器最大深度：5 层
    random_state=42,              # 随机种子：42
    verbose=0                     # 日志级别：0（静默）
)

# 打印模型参数配置信息
# 参数说明：直接访问模型属性获取参数值
print("\n" + "=" * 60)  # 打印分隔线
print("GBDT 模型参数配置（手动指定弱分类器）：")  # 打印标题
print(f"  损失函数 (loss)：{gbdt_model.loss}")  # 打印损失函数
print(f"  学习率 (learning_rate)：{gbdt_model.learning_rate}")  # 打印学习率
print(f"  弱分类器数量 (n_estimators)：{gbdt_model.n_estimators}")  # 打印树数量
print(f"  弱分类器最大深度 (max_depth)：{gbdt_model.max_depth}")  # 打印最大深度
print(f"  最小分裂样本数 (min_samples_split)：{gbdt_model.min_samples_split}")  # 打印最小分裂样本数
print(f"  叶子节点最小样本数 (min_samples_leaf)：{gbdt_model.min_samples_leaf}")  # 打印叶子最小样本数
print(f"  最大叶子节点数 (max_leaf_nodes)：{gbdt_model.max_leaf_nodes}")  # 打印最大叶子数
print(f"  最小不纯度减少 (min_impurity_decrease)：{gbdt_model.min_impurity_decrease}")  # 打印不纯度阈值
print(f"  最大特征数 (max_features)：{gbdt_model.max_features}")  # 打印最大特征数
print(f"  子采样比例 (subsample)：{gbdt_model.subsample}")  # 打印子采样比例
print(f"  剪枝参数 (ccp_alpha)：{gbdt_model.ccp_alpha}")  # 打印剪枝参数

# ============================================================
# 第七步：训练模型
# ============================================================

# 调用 fit 方法训练 GBDT 模型
# 参数说明：
#   X_train：训练集特征矩阵，形状为 (48854, 20)
#   y_train：训练集标签数组，形状为 (48854,)，值为 0 或 1
print("\n正在训练 GBDT 模型，请稍候...")  # 打印训练开始提示
gbdt_model.fit(X_train, y_train)  # 执行模型训练
print("模型训练完成！")  # 打印训练完成提示

# ============================================================
# 第八步：输出训练集精度和测试集精度
# ============================================================

# 使用训练好的模型对训练集进行预测
# 参数说明：X_train 为训练集特征矩阵；predict() 返回预测类别数组
y_train_pred = gbdt_model.predict(X_train)  # 训练集预测

# 使用训练好的模型对测试集进行预测
# 参数说明：X_test 为测试集特征矩阵；predict() 返回预测类别数组
y_test_pred = gbdt_model.predict(X_test)  # 测试集预测

# 计算训练集上的分类准确率
# 参数说明：y_train 为真实标签，y_train_pred 为预测标签
train_accuracy = accuracy_score(y_train, y_train_pred)  # 计算训练集准确率

# 计算测试集上的分类准确率
# 参数说明：y_test 为真实标签，y_test_pred 为预测标签
test_accuracy = accuracy_score(y_test, y_test_pred)  # 计算测试集准确率

# 打印训练集和测试集的精度结果
# 参数说明：格式化字符串保留 4 位小数，乘以 100 转换为百分比
print("\n" + "=" * 60)  # 打印分隔线
print("模型评估结果：")  # 打印评估标题
print(f"  训练集精度 (Train Accuracy)：{train_accuracy:.4f}（{train_accuracy * 100:.2f}%）")  # 训练集精度
print(f"  测试集精度 (Test Accuracy) ：{test_accuracy:.4f}（{test_accuracy * 100:.2f}%）")  # 测试集精度

# ============================================================
# 第九步：输入测试数据，打印预测结果
# ============================================================

# 从测试集中随机选取 5 条样本作为演示输入数据
# 参数说明：np.random.seed(42) 设置随机种子保证可复现
np.random.seed(42)  # 设置 numpy 随机种子
# 参数说明：np.random.choice 从 [0, len(X_test)) 中不放回抽取 5 个索引
sample_indices = np.random.choice(len(X_test), size=5, replace=False)  # 随机选 5 个索引

# 根据索引提取对应的特征数据和真实标签
# 参数说明：iloc[] 按位置索引提取行
sample_X = X_test.iloc[sample_indices]  # 提取 5 条样本特征
sample_y_true = y_test[sample_indices]  # 提取 5 条样本真实标签

# 使用模型对这 5 条样本进行预测
# 参数说明：sample_X 形状为 (5, 20)
sample_y_pred = gbdt_model.predict(sample_X)  # 预测类别

# 获取每条样本属于各类别的预测概率
# 参数说明：返回形状为 (5, 2) 的数组，[i][0] 为可食用概率，[i][1] 为有毒概率
sample_y_proba = gbdt_model.predict_proba(sample_X)  # 预测概率

# 定义类别名称映射字典
# 参数说明：0 → 可食用，1 → 有毒
class_names = {0: 'edible（可食用）', 1: 'poisonous（有毒）'}  # 类别映射

# 打印每条测试样本的详细预测结果
# 参数说明：enumerate 同时获取序号 i 和原始索引 idx
print("\n" + "=" * 60)  # 打印分隔线
print("输入测试数据预测结果（随机选取 5 条测试样本）：")  # 打印标题
print("-" * 60)  # 打印虚线分隔符

for i, idx in enumerate(sample_indices):  # 遍历 5 个样本
    # 获取当前样本的真实标签
    true_label = sample_y_true[i]  # 真实标签（0 或 1）
    # 获取当前样本的预测标签
    pred_label = sample_y_pred[i]  # 预测标签（0 或 1）
    # 获取当前样本的预测概率
    proba = sample_y_proba[i]  # 概率数组 [p_edible, p_poisonous]
    # 判断预测是否正确
    is_correct = "✓ 正确" if true_label == pred_label else "✗ 错误"  # 正确性标注

    # 打印当前样本的详细信息
    print(f"\n样本 {i + 1}（测试集索引：{idx}）：")  # 打印序号
    print(f"  真实标签：{class_names[true_label]}（编码值：{true_label}）")  # 真实标签
    print(f"  预测标签：{class_names[pred_label]}（编码值：{pred_label}）")  # 预测标签
    print(f"  预测概率：可食用 = {proba[0]:.4f}，有毒 = {proba[1]:.4f}")  # 各类别概率
    print(f"  预测结果：{is_correct}")  # 正确/错误标注

print("\n" + "=" * 60)  # 打印结束分隔线
print("GBDT 蘑菇分类实验完成！")  # 打印完成提示