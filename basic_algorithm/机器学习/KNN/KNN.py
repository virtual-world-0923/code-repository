import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# ===================== 配置区（可按需修改） =====================
# 本地iris.data路径
DATA_PATH = r"D:\Repositories\basic_algorithm\database\iris.data"
K_VALUE = 5
# 【二维可视化使用】选择花瓣长、花瓣宽，分类边界清晰
FEATURE_IDX = [2, 3]
# 【如需切换4维特征训练，取消下面注释，注释上面一行】
# FEATURE_IDX = [0, 1, 2, 3]

# 标签映射
label_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

# ===================== 加载数据集 =====================
try:
    # 优先读取本地iris.data文件
    col_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "species"]
    df = pd.read_csv(DATA_PATH, names=col_names)
    X_raw = df.iloc[:, :-1].values
    y_raw = df["species"].map(label_mapping).values
    print("✅ 成功读取本地 iris.data 文件")
except FileNotFoundError:
    # 本地文件不存在，使用sklearn内置数据集兜底
    print("⚠️ 本地文件未找到，自动使用sklearn内置鸢尾花数据集")
    iris = load_iris()
    X_raw = iris.data
    y_raw = iris.target

# 选取指定维度特征
X = X_raw[:, FEATURE_IDX]
y = y_raw

# ===================== 划分训练集、测试集 =====================
# stratify=y 保证各类别划分比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ===================== KNN训练 =====================
# 'euclidean'	欧氏距离	最常用，直线距离
# 'manhattan'	曼哈顿距离	城市街区距离
# 'chebyshev'	切比雪夫距离	坐标差值最大值
# 'minkowski'	闵可夫斯基距离	通用形式，搭配参数p
# 'cosine'	余弦距离	衡量向量夹角，适合高维
knn_model = KNeighborsClassifier(n_neighbors=K_VALUE,metric="euclidean")
knn_model.fit(X_train, y_train)

# 训练集精度
y_train_pred = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n训练集精度：{train_accuracy:.4f}")

# ===================== 测试集预测 & 精度 =====================
y_test_pred = knn_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集精度：{test_accuracy:.4f}")

# 找到预测错误样本索引
wrong_index = np.where(y_test != y_test_pred)[0]
print(f"测试集错误样本数量：{len(wrong_index)}")

# ===================== 新增：打印每个错误样本最近K个邻居类别 =====================
if len(wrong_index) > 0:
    print("\n========== 错误样本近邻信息 ==========")
    for err_idx in wrong_index:
        test_sample = X_test[err_idx:err_idx+1]
        # 获取最近k个邻居（返回距离、索引）
        dists, neighbor_indices = knn_model.kneighbors(test_sample)
        # 取出训练集中邻居对应的标签
        neighbor_labels = y_train[neighbor_indices[0]]
        true_label = y_test[err_idx]
        pred_label = y_test_pred[err_idx]

        print(f"\n错误测试样本下标：{err_idx}")
        print(f"样本特征：{test_sample[0]}")
        print(f"真实类别：{true_label}，模型预测类别：{pred_label}")
        print(f"最近{K_VALUE}个邻居类别：{neighbor_labels.tolist()}")
else:
    print("\n✅ 测试集无预测错误样本！")

# ===================== 可视化（仅二维特征可用！四维特征请注释此段绘图代码） =====================
if len(FEATURE_IDX) == 2:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    class_name = ["山鸢尾 setosa", "变色鸢尾 versicolor", "维吉尼亚鸢尾 virginica"]

    # 绘制各类测试样本
    for cls_id in range(3):
        mask = y_test == cls_id
        ax.scatter(
            X_test[mask, 0], X_test[mask, 1],
            c=color_list[cls_id], label=class_name[cls_id], s=70, alpha=0.75
        )

    # 错误样本：红色外圈标记
    ax.scatter(
        X_test[wrong_index, 0], X_test[wrong_index, 1],
        s=160, facecolor="none", edgecolor="red", linewidth=2, label="预测错误样本"
    )

    ax.set_xlabel("花瓣长度 petal length")
    ax.set_ylabel("花瓣宽度 petal width")
    ax.set_title(f"KNN鸢尾花分类 K={K_VALUE} | 测试准确率={test_accuracy:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
else:
    print("\n⚠️ 当前使用4维特征训练，无法二维可视化，如需绘图请切回二维特征！")