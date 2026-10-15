# ===================== 导入所需第三方库 =====================
# pandas：实现CSV文件读取、表格数据加载与处理
import pandas as pd
# numpy：构造测试样本数组、数值计算
import numpy as np
# train_test_split：数据集切分，分离训练集与测试集
from sklearn.model_selection import train_test_split
# DecisionTreeClassifier：sklearn决策树分类器（多分类任务原生支持）
from sklearn.tree import DecisionTreeClassifier
# accuracy_score：计算分类任务准确率指标
from sklearn.metrics import accuracy_score
# export_text：将训练完成的决策树导出为人类可读文本规则
from sklearn.tree import export_text

# ===================== 数据集路径定义 =====================
# r""原始字符串，避免Windows路径中反斜杠转义问题
# 红酒数据集文件路径
red_path = r"/database/wine/winequality-red.csv"
# 白酒数据集文件路径
white_path = r"/database/wine/winequality-white.csv"

# ===================== 读取红酒数据 =====================
# pd.read_csv()：读取csv文件
# filepath_or_buffer：传入文件完整路径
# sep=';'：葡萄酒数据集固定使用分号作为分隔符，不能用默认逗号
df_red = pd.read_csv(filepath_or_buffer=red_path, sep=';')

# ===================== 读取白酒数据 =====================
df_white = pd.read_csv(filepath_or_buffer=white_path, sep=';')

# ===================== 合并红酒、白酒两份数据集 =====================
# pd.concat()：表格拼接函数
# objs=[df_red, df_white] 需要合并的数据列表
# axis=0：纵向按行拼接（叠加样本）
# ignore_index=True：重新生成连续行索引，消除两份数据自带重复索引
df_total = pd.concat(objs=[df_red, df_white], axis=0, ignore_index=True)

# ===================== 分离特征矩阵X 和 标签y =====================
# drop(labels='quality', axis=1)：删除quality列，剩余全部列作为输入特征
X = df_total.drop(labels="quality", axis=1)
# 提取quality列作为标签，葡萄酒质量分为多个分数，属于多分类任务
y = df_total["quality"]

# ===================== 划分训练集、测试集 =====================
# train_test_split()：随机切分数据集
# 参数说明：
# X, y：特征与对应标签
# test_size=0.2：20%数据作为测试集，80%作为训练集
# random_state=42：固定随机种子，保证每次运行切分结果完全一致，方便复现
# stratify=y：分层抽样，保证训练/测试集中各类质量样本占比和原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===================== 初始化决策树分类器（适配sklearn1.9.0） =====================
# DecisionTreeClassifier 决策树分类模型
# criterion='gini'：节点划分标准，基尼系数；可选 "entropy" 信息熵 / "log_loss"
# max_depth=7：限制树最大生长深度，防止模型过拟合；设为None代表不限制深度
# random_state=42：固定随机种子，保证每次训练生成的决策树结构不变
# min_samples_split=2：一个内部节点最少拥有2个样本，才允许继续分裂
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42,
    min_samples_split=2
)

# ===================== 使用训练集训练决策树模型 =====================
# .fit()：模型训练核心方法，输入训练特征、训练标签
dt_model.fit(X_train, y_train)

# ===================== 预测并计算训练集、测试集准确率 =====================
# .predict()：传入特征，输出模型预测标签
y_train_predict = dt_model.predict(X_train)
# accuracy_score(真实标签,预测标签)：计算预测正确样本占总样本比例
train_accuracy = accuracy_score(y_train, y_train_predict)

# 对测试集进行预测
y_test_predict = dt_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_predict)

# 打印训练集、测试集精度，保留4位小数方便观察
print("=" * 70)
print(f"训练集准确率：{train_accuracy:.4f}")
print(f"测试集准确率：{test_accuracy:.4f}")
print("=" * 70)

# ===================== 打印决策树重要属性（理解决策树核心） =====================
print("########################## 决策树核心属性信息 ##########################")
# feature_names_in_：训练使用的所有特征名称（sklearn1.2及以上标准属性）
print(f"1. 全部特征名称 feature_names_in_：\n{dt_model.feature_names_in_}\n")

# classes_：模型识别到的全部分类标签（葡萄酒所有质量分数类别）
print(f"2. 分类标签集合 classes_：{dt_model.classes_}\n")

# n_classes_：多分类任务中类别总数量
print(f"3. 类别总数 n_classes_：{dt_model.n_classes_}\n")

# n_features_in_：训练时输入的特征总个数
print(f"4. 输入特征数量 n_features_in_：{dt_model.n_features_in_}\n")

# tree_.max_depth：训练完成后决策树实际生长的最大深度
print(f"5. 决策树真实最大深度 tree_.max_depth：{dt_model.tree_.max_depth}\n")

# tree_.node_count：整棵树所有节点总数（根节点+分裂节点+叶子节点）
print(f"6. 树总节点数量 tree_.node_count：{dt_model.tree_.node_count}\n")

# feature_importances_：各个特征的重要性权重，所有权重相加总和等于1
print("7. 各特征重要性 feature_importances_：")
# 循环配对特征名称与重要性数值，格式化输出
for feat_name, weight in zip(dt_model.feature_names_in_, dt_model.feature_importances_):
    print(f"    {feat_name:<22s} 权重：{weight:.4f}")
print()

# ===================== 文本输出完整决策树分支规则 =====================
print("########################## 决策树文本结构规则 ##########################")
# export_text：把决策树转换成直观文本
# decision_tree：训练完成的决策树模型
# feature_names：传入特征名，文本中直接显示特征名称而不是下标
tree_rule_text = export_text(decision_tree=dt_model, feature_names=dt_model.feature_names_in_)
print(tree_rule_text)

# ===================== 自定义一条葡萄酒测试样本，预测并打印决策路径【修复警告】 =====================
# 使用DataFrame构造样本，附带特征名称，消除特征名不匹配警告
# 特征顺序必须严格和原始数据集特征顺序保持一致
test_sample = pd.DataFrame(
    [[6.8, 0.26, 0.30, 2.1, 0.039, 36, 118, 0.9932, 3.27, 0.44, 10.6]],
    columns=dt_model.feature_names_in_
)

print("########################## 单条样本预测与完整决策路径 ##########################")
print(f"输入测试样本特征：\n{test_sample}\n")

# 预测样本质量等级
sample_pred_label = dt_model.predict(test_sample)
print(f"样本预测葡萄酒质量等级：{sample_pred_label[0]}")

# predict_proba：输出样本属于每一个类别的概率
sample_prob = dt_model.predict_proba(test_sample)
prob_result = dict(zip(dt_model.classes_, np.round(sample_prob[0], 3)))
for c,p in prob_result.items():
    print(f"各类别预测概率：{c} -- {p}")

# decision_path()：获取样本在决策树中经过的所有节点路径（sklearn1.9.0原生支持）
path_sparse_matrix = dt_model.decision_path(test_sample)
# 提取路径经过的节点编号
pass_node_list = path_sparse_matrix.indices
print(f"样本经过节点编号列表：{pass_node_list}\n")

print("【样本逐层决策路径推演】")
# 取出底层树结构对象
tree_struct = dt_model.tree_
# 遍历路径中每一个节点
for node_id in pass_node_list:
    # 判断是否叶子节点：叶子节点左右子节点编号相等
    if tree_struct.children_left[node_id] == tree_struct.children_right[node_id]:
        node_type = "叶子节点（最终分类节点）"
        # 获取叶子节点各类样本数量，确定最终类别
        node_sample_dist = tree_struct.value[node_id][0]
        final_class = dt_model.classes_[np.argmax(node_sample_dist)]
        print(f"节点{node_id:2d} | {node_type} → 判定类别：{final_class}")
    else:
        node_type = "分裂节点"
        # 获取当前节点用于划分的特征下标
        feat_index = tree_struct.feature[node_id]
        feat_name = dt_model.feature_names_in_[feat_index]
        # 当前节点划分阈值
        threshold = tree_struct.threshold[node_id]
        # 获取测试样本这条特征的取值
        sample_val = test_sample.iloc[0, feat_index]
        # 判断向左分支还是向右分支
        if sample_val <= threshold:
            branch_msg = f"{sample_val:.4f} ≤ {threshold:.4f} → 进入左子树"
        else:
            branch_msg = f"{sample_val:.4f} > {threshold:.4f} → 进入右子树"
        print(f"节点{node_id:2d} | {node_type} | 判断条件：{feat_name} {branch_msg}")

print("\n======================== 代码执行结束 ========================")