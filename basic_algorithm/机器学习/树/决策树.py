# ============================================================
# 导入所需的全部库和模块
# ============================================================

# 导入 numpy 库并起别名为 np，用于数值计算和数组操作
import numpy as np

# 导入 pandas 库并起别名为 pd，用于读取 CSV 格式的数据文件
import pandas as pd

# 从 sklearn.tree 模块中导入 DecisionTreeClassifier 类，用于创建决策树分类器对象
from sklearn.tree import DecisionTreeClassifier

# 从 sklearn.tree 模块中导入 export_text 函数，用于将决策树结构以文本形式打印输出
from sklearn.tree import export_text

# 从 sklearn.metrics 模块中导入 accuracy_score 函数，用于计算分类准确率
from sklearn.metrics import accuracy_score

# ============================================================
# 第一部分：加载数据集
# ============================================================

# 定义 iris.data 文件的绝对路径字符串，赋值给变量 data_path
# 该文件为 UCI 经典鸢尾花数据集，无表头，共 150 行 5 列
data_path = r"/database/iris.data"

# 定义列名列表 column_names，对应 iris.data 中的 5 列：
# 花萼长度(cm)、花萼宽度(cm)、花瓣长度(cm)、花瓣宽度(cm)、类别名称
column_names = [
    "sepal_length",   # 第 1 列：花萼长度，单位 cm
    "sepal_width",    # 第 2 列：花萼宽度，单位 cm
    "petal_length",   # 第 3 列：花瓣长度，单位 cm
    "petal_width",    # 第 4 列：花瓣宽度，单位 cm
    "class"           # 第 5 列：鸢尾花类别（setosa / versicolor / virginica）
]

# 使用 pandas 的 read_csv 函数读取数据文件，参数说明：
#   data_path    : 数据文件路径
#   header=None  : 表示文件没有表头行，不要将第一行当作列名
#   names        : 手动指定列名列表，即上面定义的 column_names
# 返回值 df 是一个 DataFrame 对象，包含全部 150 条样本
df = pd.read_csv(data_path, header=None, names=column_names)

# 打印数据集的前 5 行，用于快速检查数据是否正确加载
# head(5) 参数 5 表示取前 5 行
print("=" * 60)                          # 打印 60 个等号作为分隔线
print("【数据集前 5 行预览】")             # 打印标题提示
print(df.head(5))                        # 调用 head(5) 输出前 5 行数据
print(f"数据集总样本数：{len(df)}")       # 使用 len(df) 获取 DataFrame 行数并打印

# ============================================================
# 第二部分：准备特征矩阵 X 和标签向量 y
# ============================================================

# 定义特征列名列表 feature_cols，包含 4 个数值型特征列
# 这 4 列将作为决策树的输入特征（自变量）
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 从 DataFrame 中选取特征列，并调用 .values 将 DataFrame 转换为 numpy 二维数组
# 结果 X 的形状为 (150, 4)，即 150 个样本、4 个特征
X = df[feature_cols].values

# 从 DataFrame 中选取 "class" 列作为标签，并调用 .values 转换为一维 numpy 数组
# 结果 y 的形状为 (150,)，每个元素是字符串类别名，如 "Iris-setosa"
y = df["class"].values

# 打印特征矩阵 X 的形状，确认维度为 (150, 4)
print(f"\n特征矩阵 X 的形状：{X.shape}")   # X.shape 返回元组 (行数, 列数)

# 打印标签向量 y 的形状，确认维度为 (150,)
print(f"标签向量 y 的形状：{y.shape}")      # y.shape 返回元组 (样本数,)

# 使用 np.unique 获取 y 中所有不重复的类别名称，返回值赋给 classes
# np.unique 会自动排序，结果为 ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
classes = np.unique(y)

# 打印所有类别名称，方便后续对照叶子节点中的类别比例
print(f"类别列表：{classes}")               # 打印去重后的类别数组

# ============================================================
# 第三部分：创建并训练决策树分类器（使用全部数据训练）
# ============================================================

# 创建 DecisionTreeClassifier 决策树分类器实例，参数说明：
#   criterion="gini"   : 节点分裂的评估标准，使用基尼不纯度（另一可选值为 "entropy" 信息熵）
#   max_depth=None     : 不限制树的最大深度，让树自然生长直到所有叶子节点纯净或达到其他停止条件
#   min_samples_split=2: 内部节点再次分裂所需的最少样本数，2 为默认值
#   min_samples_leaf=1 : 每个叶子节点至少包含的样本数，1 为默认值
#   random_state=42    : 随机种子，设为 42 保证每次运行结果可复现
clf = DecisionTreeClassifier(
    criterion="gini",       # 分裂标准：基尼系数
    max_depth=3,         # 最大深度：不限制
    min_samples_split=2,    # 内部节点最小分裂样本数
    min_samples_leaf=1,     # 叶子节点最小样本数
    random_state=42         # 随机种子，确保可复现
)

# 调用 fit 方法训练决策树模型，参数说明：
#   X : 特征矩阵，形状 (150, 4)，即全部 150 个样本的 4 个特征
#   y : 标签向量，形状 (150,)，即全部 150 个样本的类别标签
clf.fit(X, y)

# 打印提示信息，表明模型训练完成
print("\n" + "=" * 60)                     # 打印分隔线
print("【决策树模型训练完成（使用全部 150 条数据）】")  # 打印训练完成提示

# ============================================================
# 第四部分：输出训练集精度
# ============================================================

# 调用 clf.predict(X) 对训练集 X 进行预测，返回预测标签数组 y_pred
# 参数 X 为全部 150 条样本的特征矩阵
y_pred = clf.predict(X)

# 调用 accuracy_score 函数计算训练集上的分类准确率，参数说明：
#   y      : 真实标签数组，形状 (150,)
#   y_pred : 预测标签数组，形状 (150,)
# 返回值为 0~1 之间的浮点数，1.0 表示完全正确
train_accuracy = accuracy_score(y, y_pred)

# 打印训练集精度，使用 f-string 格式化保留 4 位小数
print(f"训练集精度（Accuracy）：{train_accuracy:.4f}")  # 输出如 1.0000

# ============================================================
# 第五部分：输出每个叶子节点中各类别的比例
# ============================================================

# 打印分隔线和标题
print("\n" + "=" * 60)                     # 打印 60 个等号分隔线
print("【每个叶子节点中各类别的比例】")       # 打印标题

# 获取决策树内部的 Tree 对象，赋值给 tree_
# clf.tree_ 是 sklearn 中 _tree.Tree 的实例，存储了树的完整结构信息
tree_ = clf.tree_

# ★★★ 修复点 ★★★
# 直接使用前面定义的 feature_cols 列表作为特征名称
# 因为 fit() 传入的是 numpy 数组而非 DataFrame，所以 clf.feature_names_in_ 不存在
# 改为直接引用 feature_cols 即可
feature_name = feature_cols

# 遍历所有节点（节点编号从 0 到 node_count-1）
# tree_.node_count 返回树中节点的总数（包括内部节点和叶子节点）
for node_id in range(tree_.node_count):

    # 判断当前节点是否为叶子节点：
    # tree_.children_left[node_id] == -1 表示该节点没有左子节点
    # tree_.children_right[node_id] == -1 表示该节点没有右子节点
    # 两个条件同时满足说明是叶子节点
    if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:

        # 获取该叶子节点中各类别的样本数量数组
        # tree_.value[node_id] 的形状为 (1, n_classes)，需要用 [0] 取出一维数组
        # 里面是每个类别的比例
        class_ratios  = tree_.value[node_id][0]

        # 用n_node_samples取该叶子真实的整数样本总数
        # tree_.n_node_samples[node_id] 返回落入该节点的样本个数，是整数
        total_samples = tree_.n_node_samples[node_id]

        #用「比例 × 总数」还原各类别的真实计数，并四舍五入取整
        # np.round 避免浮点误差（如 47*0.979 可能得 46.9999），再 astype(int) 转整数
        class_counts = np.round(class_ratios * total_samples).astype(int)

        # 获取该叶子节点预测的类别索引：取样本数最多的类别
        # np.argmax 返回数组中最大值的索引位置
        predicted_class_idx = np.argmax(class_counts)

        # 通过索引从 classes 数组中获取预测的类别名称字符串
        predicted_class_name = classes[predicted_class_idx]

        # 打印叶子节点编号、总样本数、预测类别
        print(f"\n  叶子节点 {node_id}：")                          # 打印节点编号
        print(f"    总样本数：{total_samples}")                 # 打印该叶子的样本总数
        print(f"    预测类别：{predicted_class_name}")               # 打印预测类别名称

        # 遍历每个类别，打印该叶子节点中各类别的数量和比例
        for i in range(len(classes)):                                # i 从 0 到类别数-1
            # classes[i] 为第 i 个类别名称
            # int(class_counts[i]) 为该类别在此叶子中的样本数
            # class_ratios[i]*100 为该类别占比百分比
            print(f"    {classes[i]:20s} : {int(class_counts[i]):3d} 个  ({class_ratios[i]*100:5.1f}%)")

# ============================================================
# 第六部分：输出决策树的完整结构
# ============================================================

# 打印分隔线和标题
print("\n" + "=" * 60)                     # 打印 60 个等号分隔线
print("【决策树结构】")                      # 打印标题

# 调用 export_text 函数将决策树结构转换为可读的文本字符串，参数说明：
#   clf                : 训练好的决策树分类器对象
#   feature_names      : 特征名称列表，用于在输出中显示特征名而非 feature_0、feature_1
#   max_depth          : 显示的最大深度，设为 10 防止输出过长（可根据需要调大）
#   spacing            : 每层缩进的空格数，设为 4 使层次更清晰
#   decimals           : 数值保留的小数位数，设为 2
tree_text = export_text(
    clf,                                   # 传入训练好的分类器对象
    feature_names=feature_name,            # 传入特征名称列表（即 feature_cols）
    max_depth=10,                          # 最多显示 10 层深度
    spacing=4,                             # 每层缩进 4 个空格
    decimals=2                             # 阈值保留 2 位小数
)

# 打印决策树的文本结构
print(tree_text)                           # 输出完整的树结构文本

# ============================================================
# 第七部分：输入测试数据，打印决策路径
# ============================================================

# 打印分隔线和标题
print("=" * 60)                            # 打印 60 个等号分隔线
print("【输入测试数据，打印决策路径】")       # 打印标题

# 定义一条测试样本数据，使用列表形式：
# [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]
# 这里以 [5.1, 3.5, 1.4, 0.2] 为例（典型的 Iris-setosa 样本）
test_sample = [5.1, 3.5, 1.4, 0.2]

# 打印测试样本的原始输入值
print(f"\n测试样本输入：{test_sample}")     # 打印用户输入的 4 个特征值

# 将测试样本列表转换为 numpy 二维数组，形状为 (1, 4)
# reshape(1, -1) 参数说明：1 表示 1 行（1 个样本），-1 表示自动推断列数（此处为 4）
# sklearn 的 predict / decision_path 要求输入为二维数组
test_array = np.array(test_sample).reshape(1, -1)

# 调用 clf.predict 对测试样本进行预测，返回预测类别数组（长度为 1）
# 参数 test_array 形状为 (1, 4)
predicted_label = clf.predict(test_array)

# 打印预测结果，predicted_label[0] 取出数组中的第一个（也是唯一一个）元素
print(f"预测类别：{predicted_label[0]}")    # 输出如 Iris-setosa

# 调用 clf.decision_path 获取测试样本在树中的决策路径，参数说明：
#   test_array : 形状 (1, 4) 的测试样本数组
# 返回值为稀疏矩阵（CSR 格式），形状为 (1, node_count)
# 其中值为 1 的位置表示该样本经过了该节点
path_sparse = clf.decision_path(test_array)

# 将稀疏矩阵转换为密集数组，并取第 0 行（第 0 个样本的路径）
# .toarray() 将稀疏矩阵转为普通 numpy 数组
# [0] 取出第一个样本的路径向量
path_array = path_sparse.toarray()[0]

# 使用 np.where 获取路径向量中值为 1 的节点索引数组
# 这些索引就是测试样本从根节点到叶子节点经过的所有节点编号
node_indices = np.where(path_array == 1)[0]

# 打印决策路径标题
print(f"\n决策路径（从根节点到叶子节点）：")  # 打印路径标题

# 遍历路径上的每一个节点，逐步打印决策过程
for step, node_id in enumerate(node_indices):

    # 判断当前节点是否为叶子节点（左右子节点均为 -1）
    if tree_.children_left[node_id] == -1 and tree_.children_right[node_id] == -1:
        # 如果是叶子节点，先取该叶子的类别比例数组（归一化值）
        leaf_ratios = tree_.value[node_id][0]

        # 用 n_node_samples 取该叶子的真实整数样本总数
        leaf_total = tree_.n_node_samples[node_id]

        # 用「比例 × 总数」还原各类别真实计数，四舍五入后转 int，避免直接 astype(int) 截断成 0
        leaf_counts = np.round(leaf_ratios * leaf_total).astype(int)

        # 打印叶子节点信息：节点编号、还原后的各类别整数样本数、最终预测类别
        print(f"  步骤 {step + 1} → 到达叶子节点 [节点 {node_id}]")
        print(f"           各类别样本数：{dict(zip(classes, leaf_counts))}")   # 现在显示真实计数
        print(f"           最终预测类别：{predicted_label[0]}")

    else:
        # 如果是内部节点，获取该节点的分裂特征索引
        # tree_.feature[node_id] 返回用于分裂的特征列索引（0~3）
        feat_idx = tree_.feature[node_id]

        # 通过特征索引从 feature_name 列表中获取特征名称字符串
        feat_name = feature_name[feat_idx]

        # 获取该节点的分裂阈值
        # tree_.threshold[node_id] 返回分裂阈值，如 2.45
        threshold_val = tree_.threshold[node_id]

        # 获取测试样本在该分裂特征上的实际值
        # test_sample[feat_idx] 取出对应特征位置的数值
        sample_val = test_sample[feat_idx]

        # 判断测试样本的特征值是否小于等于阈值，决定走左子树还是右子树
        # sklearn 决策树规则：特征值 <= 阈值 → 左子树；特征值 > 阈值 → 右子树
        if sample_val <= threshold_val:
            direction = "左子树 (Yes)"     # 满足条件，走左子树
        else:
            direction = "右子树 (No)"      # 不满足条件，走右子树

        # 打印当前内部节点的分裂信息：
        # 节点编号、分裂特征名、阈值、样本实际值、走向
        print(f"  步骤 {step + 1} → 节点 {node_id}：")
        print(f"           判断条件：{feat_name} <= {threshold_val:.2f} ?")
        print(f"           样本值  ：{feat_name} = {sample_val:.2f}")
        print(f"           走向    ：{direction}")

# 打印结束分隔线
print("\n" + "=" * 60)                     # 打印 60 个等号作为结束分隔线
print("【程序运行结束】")                    # 打印结束提示