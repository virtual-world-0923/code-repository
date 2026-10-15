import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# -------------------------- 全局配置 --------------------------
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
RANDOM_STATE = 42  # 固定随机种子，保证结果可复现
FILE_PATH = r'D:\Repositories\basic_algorithm\database\iris.data'

# -------------------------- 1. 数据加载 --------------------------
# 鸢尾花数据集无表头，手动指定特征列名
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(FILE_PATH, header=None, names=column_names)

# 分离特征矩阵X和真实标签y_true
# K-means为无监督算法，训练时不使用y_true，仅用于后续有监督评估
X_raw = df.drop('class', axis=1)
y_true = df['class']
print("数据加载完成，样本量:", X_raw.shape[0], "特征数:", X_raw.shape[1])

# -------------------------- 2. 数据预处理 --------------------------
print("\n" + "="*50)
print("数据预处理阶段")

# 2.1 缺失值处理
missing_count = X_raw.isnull().sum().sum()
print(f"原始数据缺失值总数: {missing_count}")
# 采用均值填充（若缺失值多可考虑删除，鸢尾花通常无缺失）
X = X_raw.fillna(X_raw.mean())

# 2.2 重复值处理
duplicate_num = X.duplicated().sum()
print(f"重复样本数量: {duplicate_num}")
if duplicate_num > 0:
    X = X.drop_duplicates()
    # 同步对齐标签索引，保证后续评估维度一致
    y_true = y_true.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

# 2.3 异常值处理（IQR盖帽法）
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
# 将超出边界的异常值截断到边界值
X_clipped = X.clip(lower=lower_limit, upper=upper_limit, axis=1)#clip：每个单元格独立看，超了就改成边界值
print("IQR法异常值盖帽处理完成")

# 2.4 特征标准化（K-means必做，消除量纲影响）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clipped)
print("特征标准化完成，标准化后各特征均值≈0，标准差≈1")

# -------------------------- 3. 初始化方法对比 --------------------------
print("\n" + "="*50)
print("质心初始化方法对比（K=3）")
k = 3

# 方法1：k-means++ 初始化（sklearn默认策略）
kmeans_pp = KMeans(
    n_clusters=k,        # 聚类簇数
    init='k-means++',    # 初始化策略：k-means++
    n_init='auto',       # sklearn 1.9.0中'auto'对应k-means++运行1次
    max_iter=300,        # 单轮最大迭代次数
    tol=1e-4,            # 收敛阈值，质心移动小于该值则停止迭代
    random_state=RANDOM_STATE
)
labels_pp = kmeans_pp.fit_predict(X_scaled)
print(f"k-means++ 初始化 | 惯性SSE: {kmeans_pp.inertia_:.4f} | 迭代次数: {kmeans_pp.n_iter_}")

# 方法2：随机初始化
kmeans_random = KMeans(
    n_clusters=k,
    init='random',       # 初始化策略：随机选取样本作为质心
    n_init=10,           # 随机运行10次，取惯性最小的结果
    max_iter=300,
    tol=1e-4,
    random_state=RANDOM_STATE
)
labels_random = kmeans_random.fit_predict(X_scaled)
print(f"随机初始化 | 惯性SSE: {kmeans_random.inertia_:.4f} | 迭代次数: {kmeans_random.n_iter_}")

# 方法3：手动指定初始质心
# 取前3个标准化后的样本作为初始质心（仅作演示，实际可根据先验知识指定）
manual_centers = X_scaled[:k, :]
kmeans_manual = KMeans(
    n_clusters=k,
    init=manual_centers, # 传入自定义质心数组
    n_init=1,            # 手动指定质心只需运行1次
    max_iter=300,
    tol=1e-4,
    random_state=RANDOM_STATE
)
labels_manual = kmeans_manual.fit_predict(X_scaled)
print(f"手动指定质心 | 惯性SSE: {kmeans_manual.inertia_:.4f} | 迭代次数: {kmeans_manual.n_iter_}")

# -------------------------- 4. 最优K值筛选 --------------------------
print("\n" + "="*50)
print("最优K值筛选（K从2到10）")
k_range = range(2, 11)
inertia_list = []       # 肘部法则：惯性SSE
silhouette_list = []    # 轮廓系数
ch_list = []            # CH指数
db_list = []            # DB指数

for k in k_range:
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init='auto',
        random_state=RANDOM_STATE
    )
    labels = model.fit_predict(X_scaled)
    inertia_list.append(model.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, labels))
    ch_list.append(calinski_harabasz_score(X_scaled, labels))
    db_list.append(davies_bouldin_score(X_scaled, labels))

# 绘制4个指标的变化曲线
plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.plot(k_range, inertia_list, 'o-', color='#1f77b4')
plt.xlabel('聚类数量K')
plt.ylabel('惯性SSE')
plt.title('肘部法则')
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(k_range, silhouette_list, 'o-', color='#ff7f0e')
plt.xlabel('聚类数量K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法（越大越好）')
plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(k_range, ch_list, 'o-', color='#2ca02c')
plt.xlabel('聚类数量K')
plt.ylabel('CH指数')
plt.title('Calinski-Harabasz指数（越大越好）')
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(k_range, db_list, 'o-', color='#d62728')
plt.xlabel('聚类数量K')
plt.ylabel('DB指数')
plt.title('Davies-Bouldin指数（越小越好）')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 输出各指标推荐的最优K
best_k_sil = k_range[np.argmax(silhouette_list)]
best_k_ch = k_range[np.argmax(ch_list)]
best_k_db = k_range[np.argmin(db_list)]
print(f"轮廓系数推荐最优K: {best_k_sil}")
print(f"CH指数推荐最优K: {best_k_ch}")
print(f"DB指数推荐最优K: {best_k_db}")
print("结合鸢尾花数据集真实分类，最优K值为3")

# -------------------------- 5. 最终模型效果评估 --------------------------
print("\n" + "="*50)
print("最终模型（K=3）效果评估")
final_model = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init='auto',
    random_state=RANDOM_STATE
)
y_pred = final_model.fit_predict(X_scaled)

# 5.1 无监督评估指标
print("--- 无监督评估（无需真实标签） ---")
print(f"惯性SSE: {final_model.inertia_:.4f}")
print(f"轮廓系数: {silhouette_score(X_scaled, y_pred):.4f}")
print(f"CH指数: {calinski_harabasz_score(X_scaled, y_pred):.4f}")
print(f"DB指数: {davies_bouldin_score(X_scaled, y_pred):.4f}")

# 5.2 有监督评估指标（对比真实标签）
print("\n--- 有监督评估（对比鸢尾花真实分类） ---")
print(f"调整兰德指数ARI: {adjusted_rand_score(y_true, y_pred):.4f}")
print(f"归一化互信息NMI: {normalized_mutual_info_score(y_true, y_pred):.4f}")

# 输出最终聚类质心
print("\n最终聚类质心（标准化特征空间）：")
centroid_df = pd.DataFrame(final_model.cluster_centers_, columns=X.columns)
print(centroid_df.round(4))
