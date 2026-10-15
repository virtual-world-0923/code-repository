import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import matplotlib

# 方案A：使用 Windows 自带的微软雅黑（推荐）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 方案B（备选）：如果方案A不生效，尝试直接指定字体文件
# matplotlib.font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
# matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler #数据预处理,标准化
from sklearn.metrics import ( #模型评估指标
    silhouette_score, #轮廓系数：衡量簇内紧密度 vs 簇间分离度
    calinski_harabasz_score, #CH指数：簇间方差 / 簇内方差
    davies_bouldin_score, #DB指数：簇内散度 / 簇间距离
    adjusted_rand_score,#调整兰德指数：聚类结果与真实标签的一致性
    normalized_mutual_info_score,#标准化互信息：聚类与真实标签的信息共享程度
)
from sklearn.decomposition import PCA

# ============================================================
# 1. 加载数据
# ============================================================
# iris.data 无表头，手动指定列名
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(
    r'D:\Repositories\basic_algorithm\database\iris.data',
    header=None,
    names=columns
)
print("=" * 60)
print("原始数据概览：")
print(df.head())
print(f"\n数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")

# ============================================================
# 2. 数据处理
# ============================================================
print("\n" + "=" * 60)
print("【数据处理】")

# --- 2.1 缺失值处理 ---
print(f"\n[缺失值统计]\n{df.isnull().sum()}")
# 策略：删除含缺失值的行（也可用均值/中位数填充）
df = df.dropna().reset_index(drop=True)#dropna() → 删掉有缺失值的行，reset_index(drop=True) → 重置行号

# --- 2.2 异常值处理（IQR法） ---
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


def remove_outliers_iqr(data, cols, factor=1.5):
    """使用IQR法移除异常值"""
    mask = pd.Series([True] * len(data), index=data.index)
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask &= (data[col] >= lower) & (data[col] <= upper)
    return data[mask].reset_index(drop=True)


df_clean = remove_outliers_iqr(df, feature_cols)
print(f"\n[异常值处理] 移除前: {len(df)} 行 → 移除后: {len(df_clean)} 行")

# --- 2.3 分离特征和标签 ---
X = df_clean[feature_cols]
y_true = df_clean['species']  # 真实标签（用于评估对比）

# --- 2.4 标准化（K-Means基于欧氏距离，必须标准化） ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[标准化] 均值: {X_scaled.mean(axis=0).round(4)}")
print(f"[标准化] 标准差: {X_scaled.std(axis=0).round(4)}")

# ============================================================
# 3. 常用初始化质心方法
# ============================================================
print("\n" + "=" * 60)
print("【初始化质心方法】")

# sklearn 1.9.0 中 KMeans 的 init 参数支持：
#   - 'k-means++' (默认，智能选择)
#   - 'random'    (完全随机)
#   - ndarray     (自定义初始质心，shape=(n_clusters, n_features))

K = 3  # 鸢尾花有3类

# --- 方法1: k-means++ (默认，推荐) ---
# 原理：第一个质心随机选，后续质心选择离已有质心最远的点（概率加权）
km_kmeanspp = KMeans(
    n_clusters=K,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    random_state=42
)
km_kmeanspp.fit(X_scaled)
print(f"\n[方法1] k-means++  | Inertia（所有样本到其所属簇质心的距离的平方和）: {km_kmeanspp.inertia_:.4f} | 迭代次数: {km_kmeanspp.n_iter_}")

# --- 方法2: random (完全随机) ---
# 原理：从数据点中随机选取K个作为初始质心
km_random = KMeans(
    n_clusters=K,
    init='random',
    n_init=10,  #跑10次不同随机种子取最优
    max_iter=300,# 单次最多迭代次数
    tol=1e-4,    # 质心变化小于这个阈值就停
    random_state=42
)
km_random.fit(X_scaled)
print(f"[方法2] random     | Inertia:（所有样本到其所属簇质心的距离的平方和） {km_random.inertia_:.4f} | 迭代次数: {km_random.n_iter_}")


# --- 方法3: 自定义初始质心（基于先验知识/均匀网格） ---
# 场景：已知数据大致分布范围，手动指定初始位置，自己指定起点在哪
# 这里用各特征的最小值和最大值构造网格点作为初始质心
def custom_init_grid(X_data, n_clusters):
    """基于数据范围构造均匀网格初始质心"""
    mins = X_data.min(axis=0)
    maxs = X_data.max(axis=0)
    # 在范围内均匀取n_clusters个点
    centers = np.linspace(mins, maxs, n_clusters)
    return centers


custom_centers = custom_init_grid(X_scaled, K)
km_custom = KMeans(
    n_clusters=K,
    init=custom_centers,  # 传入ndarray
    n_init=1,  # 自定义质心时n_init只能为1
    max_iter=300,
    random_state=42
)
km_custom.fit(X_scaled)
print(f"[方法3] 自定义网格 | Inertia:（所有样本到其所属簇质心的距离的平方和） {km_custom.inertia_:.4f} | 迭代次数: {km_custom.n_iter_}")


# --- 方法4: 基于分位数/聚类的智能初始化 ---
# 思路：先用数据的分位数点作为初始质心
def custom_init_quantile(X_data, n_clusters):
    """基于分位数选择初始质心"""
    quantiles = np.linspace(0, 1, n_clusters + 2)[1:-1]  # 去掉0和1
    centers = np.quantile(X_data, quantiles, axis=0)
    return centers


quantile_centers = custom_init_quantile(X_scaled, K)
km_quantile = KMeans(
    n_clusters=K,
    init=quantile_centers,
    n_init=1,
    max_iter=300,
    random_state=42
)
km_quantile.fit(X_scaled)
print(f"[方法4] 分位数初始化| Inertia:（所有样本到其所属簇质心的距离的平方和） {km_quantile.inertia_:.4f} | 迭代次数: {km_quantile.n_iter_}")

# ============================================================
# 4. 选择最合适的K值
# ============================================================
print("\n" + "=" * 60)
print("【选择最优K值】")

K_range = range(2, 11)
inertias = []
silhouette_scores = []
ch_scores = []
db_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init='auto',
                max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)

    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# 打印各指标
print(f"\n{'K':<4}{'Inertia':<12}{'轮廓系数':<12}{'CH指数':<12}{'DB指数':<12}")
print("-" * 52)
for i, k in enumerate(K_range):
    print(f"{k:<4}{inertias[i]:<12.2f}{silhouette_scores[i]:<12.4f}"
          f"{ch_scores[i]:<12.2f}{db_scores[i]:<12.4f}")

best_k_silhouette = list(K_range)[np.argmax(silhouette_scores)]
best_k_ch = list(K_range)[np.argmax(ch_scores)]
best_k_db = list(K_range)[np.argmin(db_scores)]
print(f"\n轮廓系数最优K = {best_k_silhouette}")
print(f"CH指数最优K   = {best_k_ch}")
print(f"DB指数最优K   = {best_k_db}")

# --- 可视化：肘部法则 + 轮廓系数 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2)
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Inertia (SSE)')
axes[0, 0].set_title('肘部法则 (Elbow Method)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(K_range, silhouette_scores, 'rs-', linewidth=2)
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('轮廓系数法')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(K_range, ch_scores, 'g^-', linewidth=2)
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Calinski-Harabasz Index')
axes[1, 0].set_title('CH指数法 (越大越好)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(K_range, db_scores, 'md-', linewidth=2)
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Davies-Bouldin Index')
axes[1, 1].set_title('DB指数法 (越小越好)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('k_selection.png', dpi=150)
plt.show()

# ============================================================
# 5. 最终模型训练 & 效果评估
# ============================================================
print("\n" + "=" * 60)
print("【最终模型评估】(K=3)")

# 使用最优K训练最终模型
final_km = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    random_state=42
)
y_pred = final_km.fit_predict(X_scaled)

# --- 5.1 内部评估指标（无需真实标签） ---
print("\n[内部评估指标]")
print(f"  轮廓系数 (Silhouette Score):      {silhouette_score(X_scaled, y_pred):.4f}  (范围[-1,1], 越接近1越好)")
print(f"  CH指数 (Calinski-Harabasz):       {calinski_harabasz_score(X_scaled, y_pred):.2f}  (越大越好)")
print(f"  DB指数 (Davies-Bouldin):          {davies_bouldin_score(X_scaled, y_pred):.4f}  (越小越好)")
print(f"  Inertia (簇内平方和):             {final_km.inertia_:.4f}")

# --- 5.2 外部评估指标（有真实标签时） ---
print("\n[外部评估指标] (与真实标签对比)")
print(f"  调整兰德指数 (ARI):               {adjusted_rand_score(y_true, y_pred):.4f}  (范围[-1,1], 1为完美)")
print(f"  标准化互信息 (NMI):               {normalized_mutual_info_score(y_true, y_pred):.4f}  (范围[0,1], 1为完美)")

# --- 5.3 各簇统计 ---
print("\n[各簇样本数]")
for i in range(3):
    count = np.sum(y_pred == i)
    print(f"  簇 {i}: {count} 个样本")

# --- 5.4 降维可视化 ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(final_km.cluster_centers_)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.3)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', marker='X', s=250, edgecolors='black', linewidth=2, label='质心')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
plt.title('K-Means 聚类结果 (PCA降维可视化)')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_result.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("完成！")