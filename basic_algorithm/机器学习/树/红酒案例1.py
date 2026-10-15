# ============================================================================
# 决策树算法练习 —— 葡萄酒质量数据集（Wine Quality）多分类任务
# 适用版本：scikit-learn 1.9.0
# ============================================================================

# 导入 pandas 库，用于读取 CSV 文件和数据处理
import pandas as pd
# 导入 numpy 库，用于数值计算和数组操作
import numpy as np
# 从 sklearn.tree 模块导入决策树分类器类
from sklearn.tree import DecisionTreeClassifier
# 从 sklearn.tree 模块导入 export_text 函数，用于将决策树导出为可读文本格式
from sklearn.tree import export_text
# 从 sklearn.model_selection 模块导入 train_test_split 函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 从 sklearn.metrics 模块导入 accuracy_score 函数，用于计算分类准确率
from sklearn.metrics import accuracy_score
# 从 sklearn.metrics 模块导入 classification_report 函数，用于生成详细的分类报告
from sklearn.metrics import classification_report
# 从 sklearn.metrics 模块导入 confusion_matrix 函数，用于生成混淆矩阵
from sklearn.metrics import confusion_matrix
# 导入 os 模块，用于处理文件路径
import os

# ============================================================================
# 第一部分：加载数据集
# ============================================================================

# 定义数据集所在的文件夹路径
data_dir = r"/database/wine"

# 使用 os.path.join 拼接红酒数据文件的完整路径，参数为文件夹路径和文件名
red_wine_path = os.path.join(data_dir, "winequality-red.csv")

# 使用 os.path.join 拼接白酒数据文件的完整路径，参数为文件夹路径和文件名
white_wine_path = os.path.join(data_dir, "winequality-white.csv")

# 使用 pandas 的 read_csv 读取红酒数据集，参数 sep=';' 指定分隔符为分号（该数据集使用分号分隔）
df_red = pd.read_csv(red_wine_path, sep=';')

# 使用 pandas 的 read_csv 读取白酒数据集，参数 sep=';' 指定分隔符为分号
df_white = pd.read_csv(white_wine_path, sep=';')

# 为红酒数据添加一列 'wine_type'，值为 0 表示红酒，用于区分红酒和白酒
df_red['wine_type'] = 0

# 为白酒数据添加一列 'wine_type'，值为 1 表示白酒，用于区分红酒和白酒
df_white['wine_type'] = 1

# 使用 pd.concat 将红酒和白酒数据纵向拼接，参数 ignore_index=True 重置索引
df = pd.concat([df_red, df_white], ignore_index=True)

# 打印合并后数据集的形状（行数, 列数），帮助了解数据规模
print(f"合并后数据集形状: {df.shape}")

# 打印数据集的前5行，快速查看数据内容和列名
print(df.head())

# 打印目标变量 'quality' 的类别分布，了解各质量等级的样本数量
print(f"\n质量等级分布:\n{df['quality'].value_counts().sort_index()}")

# ============================================================================
# 第二部分：准备特征和标签
# ============================================================================

# 将 'quality' 列作为目标标签 y，其余所有列作为特征 X
# df.drop('quality', axis=1) 表示删除 'quality' 列，axis=1 表示按列删除
X = df.drop('quality', axis=1)

# 提取 'quality' 列作为目标变量 y，这是我们要预测的多分类标签（值为3~9）
y = df['quality']

# 获取特征名称列表，后续打印决策树结构和特征重要性时使用
feature_names = X.columns.tolist()

# 打印所有特征名称，确认输入特征
print(f"\n特征列表: {feature_names}")

# 打印特征矩阵的形状（样本数, 特征数）
print(f"特征矩阵 X 形状: {X.shape}")

# 打印标签数组的形状（样本数）
print(f"标签数组 y 形状: {y.shape}")

# ============================================================================
# 第三部分：划分训练集和测试集
# ============================================================================

# 使用 train_test_split 划分数据集
# 参数说明：
#   X: 特征矩阵
#   y: 标签数组
#   test_size=0.2: 测试集占总数据的 20%
#   random_state=42: 设置随机种子为42，保证每次运行划分结果一致，便于复现
#   stratify=y: 按标签 y 的类别比例进行分层抽样，确保训练集和测试集中各类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 打印训练集的样本数量
print(f"\n训练集样本数: {X_train.shape[0]}")

# 打印测试集的样本数量
print(f"测试集样本数: {X_test.shape[0]}")

# ============================================================================
# 第四部分：创建并训练决策树分类器
# ============================================================================

# 创建决策树分类器实例
# 参数说明：
#   criterion='gini': 使用基尼不纯度作为分裂标准（另一可选值为 'entropy' 信息熵）
#   max_depth=10: 限制树的最大深度为10，防止过拟合（None表示不限制）
#   min_samples_split=5: 内部节点再次分裂所需的最小样本数为5
#   min_samples_leaf=3: 每个叶子节点至少包含3个样本，防止过拟合
#   max_features='sqrt': 每次分裂时随机考虑 sqrt(n_features) 个特征
#   random_state=42: 设置随机种子为42，保证结果可复现
#   class_weight='balanced': 根据类别频率自动调整权重，处理类别不平衡问题
clf = DecisionTreeClassifier(
    criterion='gini',          # 分裂标准：基尼不纯度
    max_depth=10,              # 树的最大深度限制
    min_samples_split=2,       # 内部节点最小分裂样本数
    # min_samples_leaf=3,        # 叶子节点最小样本数
    # max_features='sqrt',       # 每次分裂考虑的特征数量
    # class_weight='balanced',    # 类别权重平衡
    random_state = 42  # 随机种子
)
clf.featu

# 使用训练集数据训练（拟合）决策树模型
# 参数说明：
#   X_train: 训练集特征矩阵，形状为 (n_samples, n_features)
#   y_train: 训练集标签数组，形状为 (n_samples,)
clf.fit(X_train, y_train)

# ============================================================================
# 第五部分：模型预测与精度评估
# ============================================================================

# 使用训练好的模型对训练集进行预测，返回每个样本的预测类别
y_train_pred = clf.predict(X_train)

# 使用训练好的模型对测试集进行预测，返回每个样本的预测类别
y_test_pred = clf.predict(X_test)

# 计算训练集准确率：accuracy_score(y_true, y_pred) 返回正确预测的比例
train_accuracy = accuracy_score(y_train, y_train_pred)

# 计算测试集准确率：accuracy_score(y_true, y_pred) 返回正确预测的比例
test_accuracy = accuracy_score(y_test, y_test_pred)

# 打印训练集精度，保留4位小数
print(f"\n{'='*60}")
print(f"训练集精度 (Train Accuracy): {train_accuracy:.4f}")

# 打印测试集精度，保留4位小数
print(f"测试集精度 (Test Accuracy):  {test_accuracy:.4f}")
print(f"{'='*60}")

# 打印测试集的详细分类报告
# 参数说明：
#   y_test: 真实标签
#   y_test_pred: 预测标签
#   输出包含每个类别的 precision(精确率)、recall(召回率)、f1-score(F1分数)、support(样本数)
print(f"\n测试集分类报告:")
print(classification_report(y_test, y_test_pred))

# 计算并打印混淆矩阵，行表示真实类别，列表示预测类别
# 参数说明：y_test 为真实标签，y_test_pred 为预测标签
cm = confusion_matrix(y_test, y_test_pred)

# 打印混淆矩阵
print(f"混淆矩阵:")
print(cm)

# ============================================================================
# 第六部分：打印决策树的重要属性和结构信息
# ============================================================================

print(f"\n{'='*60}")
print("决策树模型重要属性")
print(f"{'='*60}")

# 1. 树的深度：get_depth() 返回决策树从根节点到最深叶子节点的路径长度
print(f"\n【1】树的深度 (get_depth()): {clf.get_depth()}")

# 2. 叶子节点数量：get_n_leaves() 返回决策树中叶子节点（终端节点）的总数
print(f"【2】叶子节点数 (get_n_leaves()): {clf.get_n_leaves()}")

# 3. 树的总节点数：tree_.node_count 返回树中所有节点（内部节点+叶子节点）的总数
print(f"【3】总节点数 (tree_.node_count): {clf.tree_.node_count}")

# 4. 类别标签数组：classes_ 返回模型学习到的所有类别标签，按升序排列
print(f"【4】类别标签 (classes_): {clf.classes_}")

# 5. 类别数量：n_classes_ 返回分类任务中的类别总数
print(f"【5】类别数量 (n_classes_): {clf.n_classes_}")

# 6. 训练时的特征数量：n_features_in_ 返回 fit 时看到的特征数量
print(f"【6】输入特征数 (n_features_in_): {clf.n_features_in_}")

# 7. 每次分裂考虑的最大特征数：max_features_ 返回实际使用的 max_features 数值
print(f"【7】每次分裂最大特征数 (max_features_): {clf.max_features_}")

# 8. 树内部记录的最大深度：tree_.max_depth 与 get_depth() 相同，是底层 Tree 对象的属性
print(f"【8】tree_.max_depth: {clf.tree_.max_depth}")

# 9. 特征重要性：feature_importances_ 返回每个特征对分类的贡献度（基于不纯度减少量）
# 值越大表示该特征对分类越重要，所有值之和为1
print(f"\n【9】特征重要性 (feature_importances_):")
# 将特征名和重要性配对，按重要性降序排列后逐一打印
for name, importance in sorted(zip(feature_names, clf.feature_importances_),
                                key=lambda x: x[1], reverse=True):
    # 打印每个特征的名称和对应的重要性值，保留4位小数
    print(f"    {name:25s}: {importance:.4f}")

# 10. 树结构底层属性说明
print(f"\n【10】树结构底层属性 (tree_ 对象):")

# tree_.children_left: 数组，记录每个节点的左子节点索引，-1表示该节点为叶子节点
print(f"    children_left (左子节点索引数组): {clf.tree_.children_left[:10]}... (仅显示前10个)")

# tree_.children_right: 数组，记录每个节点的右子节点索引，-1表示该节点为叶子节点
print(f"    children_right (右子节点索引数组): {clf.tree_.children_right[:10]}... (仅显示前10个)")

# tree_.feature: 数组，记录每个节点用于分裂的特征索引，-2表示该节点为叶子节点
print(f"    feature (各节点分裂特征索引): {clf.tree_.feature[:10]}... (仅显示前10个)")

# tree_.threshold: 数组，记录每个节点的分裂阈值，样本特征值 <= 阈值走左子树，> 阈值走右子树
print(f"    threshold (各节点分裂阈值): {np.round(clf.tree_.threshold[:10], 4)}... (仅显示前10个)")

# tree_.impurity: 数组，记录每个节点的不纯度值（基尼系数），值越小表示节点越纯
print(f"    impurity (各节点不纯度/基尼值): {np.round(clf.tree_.impurity[:10], 4)}... (仅显示前10个)")

# tree_.n_node_samples: 数组，记录到达每个节点的训练样本数量
print(f"    n_node_samples (各节点样本数): {clf.tree_.n_node_samples[:10]}... (仅显示前10个)")

# tree_.weighted_n_node_samples: 数组，记录到达每个节点的加权样本数（考虑class_weight后的值）
print(f"    weighted_n_node_samples (加权样本数): {np.round(clf.tree_.weighted_n_node_samples[:10], 2)}... (仅显示前10个)")

# tree_.value: 数组，形状为 (n_nodes, 1, n_classes)，记录每个节点中各类别的加权样本数
print(f"    value (各节点类别分布，形状): {clf.tree_.value.shape}")

# tree_.n_classes: 每个输出对应的类别数
print(f"    n_classes (类别数): {clf.tree_.n_classes}")

# tree_.n_features: 特征数量
print(f"    n_features (特征数): {clf.tree_.n_features}")

# 11. 使用 export_text 导出决策树的文本表示
# 参数说明：
#   clf: 训练好的决策树分类器
#   feature_names: 特征名称列表，使输出更具可读性
#   max_depth=5: 只展示前5层，避免输出过长（完整树可能非常深）
#   spacing=4: 每层缩进4个空格
#   decimals=3: 数值保留3位小数
print(f"\n【11】决策树文本结构 (export_text, 展示前5层):")
tree_text = export_text(clf, feature_names=feature_names, max_depth=5, spacing=4, decimals=3)
# 打印导出的决策树文本结构
print(tree_text)

# ============================================================================
# 第七部分：代价复杂度剪枝路径（后剪枝参考）
# ============================================================================

print(f"\n{'='*60}")
print("代价复杂度剪枝路径 (Cost Complexity Pruning)")
print(f"{'='*60}")

# 调用 cost_complexity_pruning_path 计算剪枝路径
# 参数说明：
#   X_train: 训练集特征矩阵
#   y_train: 训练集标签数组
# 返回值是一个对象，包含：
#   ccp_alphas: 有效 alpha 值的递增序列，alpha 越大剪枝越激进
#   impurities: 对应每个 alpha 值时树的加权不纯度总和
pruning_path = clf.cost_complexity_pruning_path(X_train, y_train)

# 获取所有有效的 ccp_alpha 值（从小到大排列）
ccp_alphas = pruning_path.ccp_alphas

# 获取对应每个 alpha 值的加权不纯度
impurities = pruning_path.impurities

# 打印 alpha 值的数量，表示有多少个可能的剪枝点
print(f"剪枝路径中 alpha 值数量: {len(ccp_alphas)}")

# 打印前10个 alpha 值，了解剪枝强度的变化范围
print(f"前10个 ccp_alpha 值: {np.round(ccp_alphas[:10], 6)}")

# 打印前10个对应的不纯度值
print(f"前10个对应不纯度值: {np.round(impurities[:10], 6)}")

# 打印最大 alpha 值，超过此值树只剩根节点
print(f"最大 ccp_alpha 值: {ccp_alphas[-1]:.6f}")

# ============================================================================
# 第八部分：输入一条测试数据，打印决策路径
# ============================================================================

print(f"\n{'='*60}")
print("单条测试数据的决策路径")
print(f"{'='*60}")

# 从测试集中取出第一条样本作为测试数据
# X_test.iloc[[0]] 使用双层方括号保持二维形状 (1, n_features)，符合模型输入要求
test_sample = X_test.iloc[[0]]

# 获取该测试样本的真实标签值
true_label = y_test.iloc[0]

# 使用模型预测该测试样本的类别
predicted_label = clf.predict(test_sample)[0]

# 打印测试样本的各特征值
print(f"\n测试样本特征值:")
# 遍历每个特征名和对应的特征值，逐一打印
for fname, fval in zip(feature_names, test_sample.values[0]):
    # 打印特征名和特征值，特征值保留3位小数
    print(f"    {fname:25s}: {fval:.3f}")

# 打印该样本的真实标签
print(f"\n真实质量等级: {true_label}")

# 打印模型预测的标签
print(f"预测质量等级: {predicted_label}")

# 调用 decision_path 获取该样本在树中的决策路径
# 参数说明：
#   test_sample: 输入样本，形状为 (1, n_features)
# 返回值：稀疏矩阵 (CSR格式)，形状为 (1, n_nodes)
# 非零元素的位置表示该样本经过的节点索引
decision_path_matrix = clf.decision_path(test_sample)

# 将稀疏矩阵转换为密集数组，方便索引操作
# .toarray() 将 CSR 稀疏矩阵转为 numpy 二维数组
decision_path_array = decision_path_matrix.toarray()

# 获取该样本经过的所有节点索引（非零元素的列索引）
# np.flatnonzero 返回数组中所有非零元素的索引
node_indices = np.flatnonzero(decision_path_array[0])

# 打印决策路径经过的节点总数
print(f"\n决策路径经过的节点数: {len(node_indices)}")

# 打印完整的决策路径（从根节点到叶子节点）
print(f"\n决策路径详情（从根节点到叶子节点）:")
print(f"{'-'*60}")

# 遍历决策路径中的每个节点索引
for i, node_id in enumerate(node_indices):
    # 获取当前节点的分裂特征索引，-2 表示叶子节点（TREE_LEAF 常量值为 -2）
    feature_idx = clf.tree_.feature[node_id]

    # 获取当前节点的分裂阈值
    threshold = clf.tree_.threshold[node_id]

    # 获取当前节点的不纯度（基尼值）
    impurity = clf.tree_.impurity[node_id]

    # 获取到达当前节点的样本数
    n_samples = clf.tree_.n_node_samples[node_id]

    # 获取当前节点的类别分布（各类别的加权样本数）
    node_value = clf.tree_.value[node_id][0]

    # 获取当前节点预测的类别（取类别分布中样本数最多的类别）
    node_class = clf.classes_[np.argmax(node_value)]

    # 判断是否为叶子节点：feature == -2 (TREE_LEAF) 表示叶子节点
    if feature_idx == -2:
        # 叶子节点：打印最终分类结果
        print(f"  节点 {node_id} [叶子节点]")
        print(f"    不纯度(基尼): {impurity:.4f}")
        print(f"    样本数: {n_samples}")
        print(f"    类别分布: {np.round(node_value, 2)}")
        print(f"    预测类别: {node_class}")
        print(f"    >>> 最终决策: 质量等级 = {node_class}")
    else:
        # 内部节点：打印分裂条件
        # 获取分裂特征的名称
        split_feature_name = feature_names[feature_idx]

        # 获取测试样本在该分裂特征上的实际值
        sample_feature_value = test_sample.values[0][feature_idx]

        # 判断测试样本走左子树还是右子树：特征值 <= 阈值走左，> 阈值走右
        if sample_feature_value <= threshold:
            direction = "左子树 (<=)"
        else:
            direction = "右子树 (>)"

        # 打印当前节点的分裂信息
        print(f"  节点 {node_id} [内部节点]")
        print(f"    分裂条件: {split_feature_name} <= {threshold:.4f}")
        print(f"    样本特征值: {split_feature_name} = {sample_feature_value:.4f}")
        print(f"    走向: {direction}")
        print(f"    不纯度(基尼): {impurity:.4f}")
        print(f"    样本数: {n_samples}")
        print(f"    类别分布: {np.round(node_value, 2)}")
        print(f"    节点预测类别: {node_class}")
        print()

# ============================================================================
# 第九部分：使用 apply 方法验证决策路径（获取叶子节点索引）
# ============================================================================

print(f"\n{'='*60}")
print("使用 apply() 方法获取叶子节点")
print(f"{'='*60}")

# 调用 apply 方法返回每个样本最终到达的叶子节点索引
# 参数说明：test_sample 为输入样本，形状 (1, n_features)
# 返回值：数组，每个元素是对应样本到达的叶子节点编号
leaf_node_id = clf.apply(test_sample)

# 打印该测试样本最终到达的叶子节点索引
print(f"测试样本到达的叶子节点索引: {leaf_node_id[0]}")

# 获取该叶子节点的类别分布
leaf_value = clf.tree_.value[leaf_node_id[0]][0]

# 打印叶子节点的类别分布（各类别的加权样本数）
print(f"叶子节点类别分布: {np.round(leaf_value, 2)}")

# 打印叶子节点的样本数
print(f"叶子节点样本数: {clf.tree_.n_node_samples[leaf_node_id[0]]}")

# 打印叶子节点的不纯度
print(f"叶子节点不纯度(基尼): {clf.tree_.impurity[leaf_node_id[0]]:.4f}")

# ============================================================================
# 第十部分：预测概率（了解模型对各分类的置信度）
# ============================================================================

print(f"\n{'='*60}")
print("预测概率分布 (predict_proba)")
print(f"{'='*60}")

# 调用 predict_proba 获取测试样本属于每个类别的概率
# 参数说明：test_sample 为输入样本
# 返回值：形状为 (1, n_classes) 的数组，每列对应一个类别的概率
proba = clf.predict_proba(test_sample)

# 打印每个类别及其对应的预测概率
print(f"各类别预测概率:")
# 遍历类别标签和对应概率
for cls, p in zip(clf.classes_, proba[0]):
    # 打印类别和概率值，概率保留4位小数
    print(f"    质量等级 {cls}: {p:.4f}")

# ============================================================================
# 第十一部分：模型参数总结
# ============================================================================

print(f"\n{'='*60}")
print("模型超参数设置总结")
print(f"{'='*60}")

# 调用 get_params() 获取模型的所有超参数设置，返回字典
params = clf.get_params()

# 遍历并打印所有超参数
for key, value in params.items():
    # 打印参数名和参数值
    print(f"    {key:25s}: {value}")

# ============================================================================
# 程序结束
# ============================================================================
print(f"\n{'='*60}")
print("决策树练习完成！")
print(f"{'='*60}")