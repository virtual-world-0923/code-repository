import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import database.read_data as rd

# 1. 加载数据集，截取前100条二分类
X_raw ,y_raw = rd.ReadIris().binary_classification()

# 2. 特征标准化
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw)

# 3. 训练sklearn逻辑回归模型
model = LogisticRegression(fit_intercept=True, max_iter=10000)
model.fit(X_norm, y_raw)

# 4. 训练集预测与评估
y_pred = model.predict(X_norm)
acc = accuracy_score(y_raw, y_pred)
print(f"Train Accuracy: {acc:.4f}")
print(f"Model Coef (4 features): {model.coef_[0]}")
print(f"Model Intercept (b): {model.intercept_[0]}")

# 5. 自定义4特征输入预测函数
def predict_sample(raw_feat):
    feat = np.array(raw_feat).reshape(1, -1)
    feat_norm = scaler.transform(feat)
    pred = model.predict(feat_norm)[0]
    prob = model.predict_proba(feat_norm)[0]
    cls = "Setosa(0)" if pred == 0 else "Versicolor(1)"
    return cls, prob

# 测试样本
test1 = [5.0, 3.4, 1.5, 0.2]
test2 = [6.1, 2.8, 4.2, 1.3]
c1, p1 = predict_sample(test1)
c2, p2 = predict_sample(test2)
print(f"\nSample {test1}: Pred={c1}, Prob[0,1]={p1.round(4)}")
print(f"Sample {test2}: Pred={c2}, Prob[0,1]={p2.round(4)}")

# 6. 绘图：选取第1、2特征可视化
f1, f2 = 1, 3
X_plot = X_norm[:, [f1, f2]]

# 生成网格点绘制决策边界
x1_min, x1_max = X_plot[:, 0].min() - 0.6, X_plot[:, 0].max() + 0.6
x2_min, x2_max = X_plot[:, 1].min() - 0.6, X_plot[:, 1].max() + 0.6
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.03),
                       np.arange(x2_min, x2_max, 0.03))
grid = np.c_[xx1.ravel(), xx2.ravel()]

# 补齐4维特征，其余维度用均值填充
grid_4d = np.zeros((grid.shape[0], 4))
grid_4d[:, f1] = grid[:, 0]
grid_4d[:, f2] = grid[:, 1]
for i in range(4):
    if i != f1 and i != f2:
        grid_4d[:, i] = np.mean(X_norm[:, i])

# 网格预测
grid_pred = model.predict(grid_4d).reshape(xx1.shape)

# 绘制画布
plt.figure(figsize=(8, 6))
# 决策区域底色
plt.contourf(xx1, xx2, grid_pred, alpha=0.25, cmap="coolwarm")

# 真实两类样本
mask0 = y_raw == 0
mask1 = y_raw == 1
plt.scatter(X_plot[mask0.ravel(), 0], X_plot[mask0.ravel(), 1], c="#2266ff", s=45, label="True Setosa(0)")
plt.scatter(X_plot[mask1.ravel(), 0], X_plot[mask1.ravel(), 1], c="#dd2222", s=45, label="True Versicolor(1)")

# 预测错误样本：黑色菱形区分
err_mask = y_pred != y_raw.ravel()
plt.scatter(X_plot[err_mask.ravel(), 0], X_plot[err_mask.ravel(), 1], c="black", marker="D", s=70, label="Misclassified Points")

plt.xlabel(f"Feature {f1+1}: (Normalized)")
plt.ylabel(f"Feature {f2+1}: (Normalized)")
plt.title("Sklearn Logistic Regression Decision Boundary")
plt.legend()
plt.grid(alpha=0.3)
plt.show()