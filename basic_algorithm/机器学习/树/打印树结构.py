# 导入NumPy库，用于数值计算和数组操作
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# 从sklearn.tree模块导入决策树分类器类
from sklearn.tree import DecisionTreeClassifier, export_text


def print_tree_info(clf, X_train, y_train):
    try: #如果训练数据是DataFrame，树会有feature_names_in_，传入数组，没有属性
        class_labels_names = clf.classes_ #数字还是字符串取决于你传入的y （回归树没有）
        feature_names = clf.feature_names_in_ #是否有这个属性取决于你传入的X类型
    except Exception:
        class_names = np.unique(y)
        class_labels_names = class_names

        input_feature_num = X.shape[1] #特征个数
        column_names = [f'col_{i+1}' for i in range(input_feature_num)]
        feature_names = column_names

    #判断clf是森林还是树
    if hasattr(clf, 'estimators_'): #森林
        print_forest(clf, X_train, y_train,feature_names,class_labels_names)
    else: #单颗树

        print_single(clf, X_train, y_train,feature_names,class_labels_names)

def print_forest(fc, X, y,feature_names,class_labels_names):
    print(f"{'🌳🌳🌳🌳🌳🌳🌳 森林':^100}")
    # n_estimators：森林中决策树总棵数
    total_tree_count = fc.n_estimators
    print(f"森林内决策树总数量：{total_tree_count}")

    print(f"全部特征名称：{list(feature_names)}")

    print(f"分类标签集合：{list(class_labels_names)}")

    class_num = len(class_labels_names)
    print(f"类别总数：{class_num}")

    single_tree = fc.estimators_[0] #第0颗树
    print_single(single_tree,X,y,feature_names,class_labels_names)

def print_single(single_tree, X, y,feature_names,class_labels_names):
    print(f"{'🌳 单颗树':^100}")

    # 获取决策树的核心数据结构
    tree = single_tree.tree_
    # 获取树的总节点数
    n_nodes = tree.node_count
    print(f"总节点数：{n_nodes}")
    print(f"总样本数: {len(X)}")
    print(f"特征名称: {feature_names},")
    print(f"标签类别名称: {class_labels_names},")
    tree_max_depth = single_tree.get_depth() #树实际生长最大深度
    print(f"树最大深度：{tree_max_depth}")
    print("\n【树完整文本结构】")
    # 导出文本可视化决策树
    tree_text = export_text(
        decision_tree=single_tree, #目标决策树对象
        feature_names=list(feature_names), #特征名称列表
        show_weights=True #显示节点内各类样本数量
    )
    print(tree_text)
    print("-" * 80)

    print(f"{'🍃 决策树叶子节点统计':^100}")
    # 打印叶子节点总数
    # 收集所有叶子节点（左子节点等于右子节点的节点，都等于-1的情况，如果有分裂，左右节点ID就会不同）
    leaf_nodes = [i for i in range(n_nodes) if tree.children_left[i] == tree.children_right[i]]
    print(f"叶子节点总数: {len(leaf_nodes)}")
    print("-" * 100)

    # 返回每个样本在树中所处的叶子节点的索引
    if isinstance(X, pd.DataFrame):#因为森林训练单颗树的时候把DataFrame 转成 NumPy 数组了，这里X是DataFrame，会报警告
        X = X.to_numpy()
    leaf_indices = single_tree.apply(X) #这里X需要arraynp,不然会报一个警告

    # 遍历每个叶子节点，enumerate从1开始编号（更自然）
    for idx, node_id in enumerate(leaf_nodes, 1):
        # 打印叶子节点标题，包含序号和节点ID
        print(f"\n【第{idx}个叶子节点 】节点ID: {node_id}")
        # 打印分隔线
        print("-" * 50)

        # 从tree获取数据
        n_raw = tree.n_node_samples[node_id]  # 当前节点样本数
        n_weighted = tree.weighted_n_node_samples[node_id]  # 加权样本总数
        weighted_ratios = tree.value[node_id][0]  # 加权占比
        weighted_counts = weighted_ratios * n_weighted  # 计算各类别加权数量

        # 从原始数据统计该节点的真实分布
        mask = (leaf_indices == node_id)  # 创建布尔掩码，标记属于当前节点的样本
        node_y = y[mask]  # 获取当前节点的所有标签

        # 统计各类别原始数量
        raw_counts = np.array([np.sum(node_y == cna) for cna in class_labels_names])
        # 计算各类别原始占比
        raw_ratios = raw_counts / n_raw

        # 打印样本数信息
        print(f"📊 样本数:")
        print(f"   - 当前节点原始（真实）样本总数: {n_raw}")  # 显示真实样本个数
        print(f"   - 当前节点加权样本总数: {n_weighted}")  # 显示加权样本总数（保留2位小数）

        # 打印各类别分布信息
        print(f"\n📈 各类别分布:")
        # 打印表头（类别、原始数量、原始占比、加权数量、加权占比）
        print(f"   {'类别':<12} {'原始数量':<12} {'原始占比':<12} {'加权数量':<12} {'加权占比':<12}")
        # 打印分隔线
        print(f"   {'-' * 60}")

        # 遍历每个类别，打印详细信息
        for i, name in enumerate(class_labels_names):
            # 格式化并打印每个类别的统计信息
            # raw_counts[i]: 原始数量（整数）
            # raw_ratios[i]: 原始占比（百分比）
            # weighted_counts[i]: 加权数量（浮点数）
            # weighted_ratios[i]: 加权占比（百分比）
            print(f"   {name:<12} {raw_counts[i]:<12.0f} {raw_ratios[i]:<12.2%} "f"{weighted_counts[i]:<12.2f} {weighted_ratios[i]:<12.2%}")
            total_from_counts = np.sum(raw_counts)

        if total_from_counts != n_raw:
            print(f"⚠️⚠️⚠️⚠️⚠️ 警告：计数不一致 (tree={n_raw}, x原始统计={total_from_counts})")

        # 判断是否使用了样本权重
        # np.allclose() 比较两个浮点数数组是否在容差范围内相等
        if np.allclose(n_weighted, n_raw):
            # 如果加权样本数等于原始样本数（在浮点误差范围内），说明未使用权重
            print(f"\n   ✅ 未使用样本权重（原始=加权）")
        else:
            # 否则说明使用了权重
            print(f"\n   ⚠️  使用了样本权重（原始≠加权）")

    # 打印结束分隔线
    print("\n" + "=" * 100)

def tree_decision_path(single_tree, test_sample,feature_names):
    if isinstance(test_sample, pd.DataFrame):
        test_sample = test_sample.to_numpy()
    # decision_path：获取样本在决策树上经过的节点路径，返回稀疏矩阵
    path_sparse = single_tree.decision_path(test_sample)
    # indices取出途经的所有节点编号
    visited_nodes = path_sparse.indices
    print(f"样本途经节点编号列表：{visited_nodes.tolist()}")
    print("\n【决策路径逐层判断条件】")

    # tree_.feature：各节点分裂特征索引
    split_feat = single_tree.tree_.feature
    # tree_.threshold：各节点分裂阈值
    split_thresh = single_tree.tree_.threshold

    # 遍历路径节点，逐层打印判断规则
    for nid in visited_nodes:
        f_idx = split_feat[nid]
        th = split_thresh[nid]
        if f_idx == -2:
            print(f"节点{nid}：到达叶子节点，停止分裂")
        else:
            feat_name = feature_names[f_idx]
            print(f"节点{nid}：判断特征【{feat_name}】，分裂阈值 = {th:.3f}")

# ============ 使用示例 ============
if __name__ == "__main__":
    # 从sklearn.datasets导入生成分类数据集的函数
    from sklearn.datasets import make_classification
    # 从sklearn.model_selection导入数据集划分工具（虽然代码中未使用，但保持导入）
    from sklearn.model_selection import train_test_split

    # 生成示例数据
    # n_samples=200: 生成200个样本
    # n_features=4: 4个特征
    # n_informative=3: 3个特征对分类有用
    # n_redundant=1: 1个特征与其他特征相关（冗余）
    # n_classes=2: 二分类问题
    # weights=[0.3, 0.7]: 类别不平衡，类别0占30%，类别1占70%
    # random_state=42: 固定随机种子，确保结果可重复
    X, y = make_classification(
        n_samples=10,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.3, 0.7],
        random_state=42
    )
    # X = pd.DataFrame(X)

    # 示例1：不加权的决策树
    # 打印示例标题
    print("示例1: 不加权的决策树")
    # 创建决策树分类器，最大深度为3，固定随机种子
    clf1 = DecisionTreeClassifier(max_depth=3, random_state=42)
    # 训练模型（未使用任何权重参数）
    clf1.fit(X, y)
    # 调用简化版函数，打印叶子节点统计信息
    print_tree_info(clf1, X, y)

    # # 示例2：使用 class_weight 的决策树
    # print("示例2: 使用 class_weight='balanced' 的决策树")
    # # 创建决策树分类器，设置class_weight='balanced'自动平衡类别权重
    # clf2 = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    # # 训练模型（使用类别权重自动平衡）
    # clf2.fit(X, y)
    # # 打印叶子节点统计信息
    # print_tree_info(clf2, X, y)
    #
    # # 示例3：使用 sample_weight 的决策树
    # print("示例3: 使用 sample_weight（少数类权重=5）的决策树")
    # # 创建决策树分类器（不设置class_weight）
    # clf3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    # # 创建样本权重数组，所有样本初始权重为1
    # sample_weights = np.ones(len(y))
    # # 将类别0（少数类）的样本权重设为5（给少数类更大的重要性）
    # sample_weights[y == 0] = 5
    # # 训练模型，传入sample_weight参数
    # clf3.fit(X, y, sample_weight=sample_weights)
    # # 打印叶子节点统计信息
    # print_tree_info(clf3, X, y)