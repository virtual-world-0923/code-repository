import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score,
                             homogeneity_score, completeness_score,
                             v_measure_score)

# ============================================================
# 1. 加载数据
# ============================================================
data_path = r'D:\Repositories\basic_algorithm\database\iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(data_path, header=None, names=column_names)

# ============================================================
# 2. 数据探索与预处理
# ============================================================
print("数据基本信息：")
print(df.info())
print("\n前5行：")
print(df.head())

# 2.1 空值检查
print("\n空值统计：")
print(df.isnull().sum())
# 若存在，按情况处理：df.dropna(inplace=True) 或 df.fillna(...)

# 2.2 异常值检测与处理
# 先提取特征（数值型），标签稍后再用
df_features = df.drop('class', axis=1)
print(f"\n原始样本数: {len(df_features)}")

# 箱线图可视化
plt.figure(figsize=(12, 4))
df_features.boxplot()
plt.title('Boxplot of Iris Features (before outlier removal)')
plt.show()

# 选择异常值处理方法：'zscore'、'iqr' 或 None
outlier_method = 'iqr'   # 可改为 'iqr' 或 None

if outlier_method == 'zscore':
    # Z-score 方法，阈值设为 3
    z_scores = np.abs(stats.zscore(df_features)) #zscore：距离均值有多少个标准差
    # 按行判断，只要该行有任意一个特征 Z-score > 3，即视为异常
    outlier_mask = (z_scores > 3).any(axis=1)
    df_clean = df[~outlier_mask]
    print(f"Z-score (阈值=3) 剔除异常值 {outlier_mask.sum()} 个样本")

elif outlier_method == 'iqr':
    # IQR 方法
    Q1 = df_features.quantile(0.25)
    Q3 = df_features.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 判断每个样本是否超出任何一个特征的范围
    outlier_mask = ((df_features < lower_bound) | (df_features > upper_bound)).any(axis=1)
    df_clean = df[~outlier_mask]
    print(f"IQR 剔除异常值 {outlier_mask.sum()} 个样本")

else:
    df_clean = df.copy()
    print("未进行异常值处理")

# 重新分离特征和标签
df_features_clean = df_clean.drop('class', axis=1)
y_true = df_clean['class']
print(f"处理后样本数: {len(df_clean)}")

# 2.3 检查数据类型，确保均为数值
print("\n特征数据类型：")
print(df_features_clean.dtypes)

# 2.4 标准化（必须做，注意在剔除异常值之后进行）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features_clean)

# ============================================================
# 3. 演示不同初始化策略（k-means++ 与 random）
# ============================================================
init_methods = ['k-means++', 'random']
for init in init_methods:
    kmeans = KMeans(n_clusters=3, init=init, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"\n初始化策略: {init}")
    print(f"轮廓系数: {sil:.4f}")

# 也可以传入自定义初始质心（示例：随机选3个点）
rng = np.random.RandomState(42)
custom_centers = X_scaled[rng.choice(X_scaled.shape[0], 3, replace=False)]
kmeans_custom = KMeans(n_clusters=3, init=custom_centers, n_init=1, random_state=42)
labels_custom = kmeans_custom.fit_predict(X_scaled)
print(f"\n初始化策略: 自定义质心（随机选取）")
print(f"轮廓系数: {silhouette_score(X_scaled, labels_custom):.4f}")

# ============================================================
# 4. 选择合适的 K 值
# ============================================================
inertias = []
silhouettes = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# 绘制肘部法则图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')

# 绘制轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'ro-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.tight_layout()
plt.show()

best_k_elbow = 3   # 观察肘部图（实际可编程找拐点）
best_k_sil = K_range[np.argmax(silhouettes)]
print(f"\n肘部法则建议 K = {best_k_elbow}, 轮廓系数建议 K = {best_k_sil}")

# ============================================================
# 5. 最终聚类及效果评估
# ============================================================
final_k = 3
kmeans_final = KMeans(n_clusters=final_k, init='k-means++', n_init='auto', random_state=42)
labels_final = kmeans_final.fit_predict(X_scaled)

# 内部评估：轮廓系数
sil_final = silhouette_score(X_scaled, labels_final)
print(f"\n最终模型 (K={final_k}) 轮廓系数: {sil_final:.4f}")

# 外部评估（使用真实标签）
ari = adjusted_rand_score(y_true, labels_final)
nmi = normalized_mutual_info_score(y_true, labels_final)
homo = homogeneity_score(y_true, labels_final)
comp = completeness_score(y_true, labels_final)
v_meas = v_measure_score(y_true, labels_final)

print(f"调整兰德指数 (ARI): {ari:.4f}")
print(f"互信息 (NMI): {nmi:.4f}")
print(f"同质性: {homo:.4f}")
print(f"完整性: {comp:.4f}")
print(f"V-measure: {v_meas:.4f}")

# ============================================================
# 6. 聚类中心与标签分布（可选）
# ============================================================
centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=df_features_clean.columns)
print("\n聚类中心（原始尺度）：")
print(centers_df)

pred_counts = pd.Series(labels_final).value_counts().sort_index()
print("\n各簇样本数：")
print(pred_counts)