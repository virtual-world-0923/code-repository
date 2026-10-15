# ===================== 导入依赖库 =====================
# pandas：读取本地iris.data数据集，构造DataFrame（携带特征名）
import pandas as pd
# numpy：数组处理
import numpy as np
# DecisionTreeClassifier：sklearn决策树分类器
from sklearn.tree import DecisionTreeClassifier
# accuracy_score：计算模型准确率
from sklearn.metrics import accuracy_score
# export_text：导出决策树文本结构
from sklearn.tree import export_text
# 屏蔽特征名称匹配的UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# ---------------------- 1. 读取本地鸢尾花数据集 ----------------------
# 指定本地iris.data文件路径
file_path = r"/database/iris.data"
# pd.read_csv读取csv格式文件
# filepath_or_buffer：文件路径
# header=None：原始文件没有表头，不将第一行作为列名
# names：手动定义4个特征列 + 1个标签列
df = pd.read_csv(
    filepath_or_buffer=file_path,
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
)

# 分离特征矩阵X、标签y
# iloc[:, 0:4] 选取全部行，0~3列，4个特征
X = df.iloc[:, 0:4]
# iloc[:, 4] 选取全部行第4列，分类标签（字符串：Iris-setosa等）
y = df.iloc[:, 4]

# ---------------------- 2. 初始化决策树模型 ----------------------
# DecisionTreeClassifier 决策树分类器
# criterion="entropy"：使用信息熵作为划分准则，可选gini（基尼系数）
# random_state=42：固定随机种子，保证每次运行树结构完全一致
# max_depth=None：不限制树深度，充分生长
dt_clf = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    max_depth=3
)

# ---------------------- 3. 使用全部数据训练模型 ----------------------
# fit：模型训练，直接传入字符串标签y，内部自动编码
dt_clf.fit(X, y)

# ---------------------- 4. 输出训练集准确率 ----------------------
# predict：基于训练特征预测标签
y_pred = dt_clf.predict(X)
# accuracy_score：对比真实标签与预测标签，计算准确率
train_acc = accuracy_score(y, y_pred)
print("="*60)
print(f"【训练集准确率】：{train_acc:.4f}")
print("="*60)

# ---------------------- 5. 输出决策树文本结构 ----------------------
# export_text 将决策树导出为人类可读文本
# feature_names=list(X.columns) 在文本中显示特征名称
tree_text = export_text(dt_clf, feature_names=list(X.columns))
print("【决策树完整文本结构】")
print(tree_text)
print("="*60)

# ---------------------- 6. 遍历所有叶子节点，输出各类别数量与占比 ----------------------
# tree_ 获取底层树结构对象
tree = dt_clf.tree_
# children_left == -1 代表当前节点没有左子节点
leaf_mask = tree.children_left == -1 #决策树是 CART算法，严格二分类，不可能出现一个分叉，所以左边没有，右边一定也没有。就像问一个树是否大于一个值，只有两个答案，大于或者小于
# 获取所有叶子节点编号
leaf_indices = np.nonzero(leaf_mask)[0]

print("【所有叶子节点样本分布与类别占比】")
# 循环遍历每一个叶子节点
for node_id in leaf_indices:
    # 当前叶子节点总样本数量
    total_samples = tree.n_node_samples[node_id]
    # 获取节点内各类样本数量，tree.value维度 [node,1,各个class比例]
    class_ratios = tree.value[node_id][0]
    # 获取类别原始字符串名称
    class_names = dt_clf.classes_

    class_counts = np.round(class_ratios * total_samples).astype(int)
    predicted_class_idx = np.argmax(class_counts)
    predicted_class_name = class_names[predicted_class_idx]

    print(f"\n  叶子节点 {node_id}：")  # 打印节点编号
    print(f"    总样本数：{total_samples}")  # 打印该叶子的样本总数
    print(f"    预测类别：{predicted_class_name}")

    print(f"叶子节点编号: {node_id}，节点总样本数: {total_samples}")
    # 遍历每个类别计算占比，分母使用当前叶子节点样本总数
    for cls_name, ratio in zip(class_names, class_ratios):
        print(f"    {cls_name}：数量={ratio * total_samples}，占比={ratio:.4f}")
print("="*60)

# ---------------------- 7. 测试样本预测 + 打印完整决策路径 ----------------------
# 使用DataFrame构造测试样本，携带特征名称，消除sklearn特征名警告
test_sample = pd.DataFrame(
    [[5.4, 3.9, 1.7, 0.4]],
    columns=list(X.columns)
)

# predict_proba：输出各类别预测概率
pred_proba = dt_clf.predict_proba(test_sample)
# predict：输出预测类别字符串
pred_label = dt_clf.predict(test_sample)
# decision_path：获取样本经过的节点路径（稀疏矩阵）
path_matrix = dt_clf.decision_path(test_sample)

print("【测试样本信息】")
print(f"测试特征：{test_sample.values[0]}")
print(f"预测类别：{pred_label[0]}")
print(f"各类别预测概率：{dict(zip(dt_clf.classes_, pred_proba[0].round(4)))}")
print("经过节点编号列表：", path_matrix.indices.tolist())

print("\n【逐层决策判断路径】")
current_node = 0
feature_names_list = list(X.columns)

# 循环逐层遍历决策路径
while True:
    # 判断是否抵达叶子节点
    if tree.children_left[current_node] == -1:
        print(f"到达叶子节点 {current_node}，最终预测类别：{pred_label[0]}")
        break
    # 当前节点划分使用的特征下标
    split_feature_idx = tree.feature[current_node]
    # 节点划分阈值
    threshold = tree.threshold[current_node]
    feature_name = feature_names_list[split_feature_idx]
    # 取出测试样本对应特征数值
    feat_value = test_sample.values[0][split_feature_idx]

    # 判断走向左子树/右子树
    if feat_value <= threshold:
        print(f"节点{current_node}: {feature_name} ≤ {threshold:.3f} → 向左分支")
        current_node = tree.children_left[current_node]
    else:
        print(f"节点{current_node}: {feature_name} > {threshold:.3f} → 向右分支")
        current_node = tree.children_right[current_node]