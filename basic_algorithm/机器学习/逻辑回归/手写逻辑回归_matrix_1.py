import numpy as np
import matplotlib.pyplot as plt
import database.read_data as rd

# -------------------------- 独立基础函数 --------------------------
# 1. 激活函数 sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. 损失函数 交叉熵
def compute_loss(y, y_hat):
    m = y.shape[0]
    eps = 1e-8
    loss = -(1/m) * np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    return loss

# 3. 梯度计算（矩阵形式，w包含偏置项）
def compute_gradient(X, y, y_hat):
    m = X.shape[0]
    dw = (1/m) * X.T @ (y_hat - y)
    return dw

# 4. 梯度下降更新参数
def gradient_descent_update(w, dw, lr):
    w = w - lr * dw
    return w

# -------------------------- 训练函数 --------------------------
def train_logistic(X, y, lr=0.1, epochs=12000):
    m, n = X.shape
    w = np.zeros((n, 1))
    loss_record = []
    for i in range(epochs):
        z = X @ w
        y_hat = sigmoid(z)
        loss = compute_loss(y, y_hat)
        loss_record.append(loss)
        dw = compute_gradient(X, y, y_hat)
        w = gradient_descent_update(w, dw, lr)
        if i % 1000 == 0:
            print(f"Iter {i:5d} | Loss: {loss:.4f}")
    return w, loss_record

# -------------------------- 预测函数 --------------------------
def predict(X, w):
    z = X @ w
    prob = sigmoid(z)
    pred = (prob > 0.5).astype(int)
    return pred, prob

# -------------------------- 数据加载与预处理 --------------------------


raw_X ,raw_y = rd.ReadIris().binary_classification()

# 标准化
X_mean = np.mean(raw_X, axis=0)
X_std = np.std(raw_X, axis=0)
X_norm = (raw_X - X_mean) / X_std

# 核心：第一列插入全1，将偏置b融合进权重w
m = X_norm.shape[0]
X_train = np.hstack([np.ones((m, 1)), X_norm])

# -------------------------- 模型训练 --------------------------
w_trained, loss_hist = train_logistic(X_train, raw_y, lr=0.1, epochs=12000)
y_pred, prob_all = predict(X_train, w_trained)
acc = np.mean(y_pred == raw_y)
print(f"\nTrain Accuracy: {acc:.4f}")
print(f"Weights w (w0=b + 4 feature weights):\n{w_trained}")

# -------------------------- 自定义样本预测演示 --------------------------
def predict_input(feat4):
    feat = np.array(feat4).reshape(1, -1)
    feat_norm = (feat - X_mean) / X_std
    x_input = np.hstack([np.ones((1,1)), feat_norm])
    pred, p = predict(x_input, w_trained)
    cls = "Setosa(0)" if pred[0,0]==0 else "Versicolor(1)"
    return cls, p[0,0]

test1 = [5.0, 3.4, 1.5, 0.2]
test2 = [6.1, 2.8, 4.2, 1.3]
c1, p1 = predict_input(test1)
c2, p2 = predict_input(test2)
print(f"\nSample {test1}: Pred={c1}, Prob={p1:.4f}")
print(f"Sample {test2}: Pred={c2}, Prob={p2:.4f}")

# -------------------------- 绘图：选取2个特征可视化 --------------------------
# 选第1、2个原始特征作图（标准化后）
f1, f2 = 1, 2
# 原始标准化二维数据（不含全1列）
plot_x = X_norm[:, [f1, f2]]
y_flat = raw_y.flatten()
pred_flat = y_pred.flatten()

# 生成网格绘制决策边界
x1_min, x1_max = plot_x[:,0].min()-0.6, plot_x[:,0].max()+0.6
x2_min, x2_max = plot_x[:,1].min()-0.6, plot_x[:,1].max()+0.6
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.03),
                       np.arange(x2_min, x2_max, 0.03))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# 补齐4维特征，其余特征填充均值，拼接全1列
grid_4d = np.zeros((grid_points.shape[0], 4))
grid_4d[:, f1] = grid_points[:,0]
grid_4d[:, f2] = grid_points[:,1]
for idx in range(4):
    if idx != f1 and idx != f2:
        grid_4d[:, idx] = np.mean(X_norm[:, idx])
grid_full = np.hstack([np.ones((grid_4d.shape[0],1)), grid_4d])
grid_pred, _ = predict(grid_full, w_trained)
grid_pred = grid_pred.reshape(xx1.shape)

# 画布1：分类边界+真实样本+预测标记
plt.figure(figsize=(8,6))
# 分割底色
plt.contourf(xx1, xx2, grid_pred, alpha=0.25, cmap="coolwarm")

# 真实两类样本
mask0 = (y_flat == 0)
mask1 = (y_flat == 1)
plt.scatter(plot_x[mask0,0], plot_x[mask0,1], c="#2266ff", s=45, label="True Setosa(0)")
plt.scatter(plot_x[mask1,0], plot_x[mask1,1], c="#dd2222", s=45, label="True Versicolor(1)")

# 预测错误样本：黑色菱形，单独区分
err_mask = (pred_flat != y_flat)
plt.scatter(plot_x[err_mask,0], plot_x[err_mask,1], c="black", marker="D", s=70, label="Misclassified Points")

plt.xlabel(f"Feature {f1+1} (Normalized)")
plt.ylabel(f"Feature {f2+1} (Normalized)")
plt.title("Logistic Regression Decision Boundary (4D Train, 2D Visual)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 画布2：损失曲线
plt.figure(figsize=(7,4))
plt.plot(loss_hist, c="#3377cc")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss Curve")
plt.grid(alpha=0.3)
plt.show()