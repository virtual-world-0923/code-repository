import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 只保留Windows自带黑体
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
from sklearn.datasets import load_iris
import database.read_data as rd


def load_iris_data():
    x, y = rd.ReadIris().binary_classification();
    return x, y


# ===================== 1. 基础工具函数（独立拆分） =====================
# 1.1 Sigmoid激活函数（单独方法）
def sigmoid(z):
    """激活函数"""
    return 1 / (1 + np.exp(-z))


# 1.2 交叉熵损失函数（单独方法）
def compute_loss(y, y_hat):
    """二分类交叉熵损失"""
    m = y.shape[0]
    loss = -(1 / m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
    return loss


# 1.3 梯度计算（单独梯度求解）
def compute_gradient(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db


# 1.4 梯度下降参数更新（单独梯度下降方法）
def gradient_descent_update(w, b, dw, db, lr):
    w -= lr * dw
    b -= lr * db
    return w, b


# ===================== 2. 逻辑回归训练封装 =====================
def logistic_regression_train(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    # 初始化权重偏置
    w = np.zeros((n, 1))
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        # 前向传播
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        # 计算损失
        loss = compute_loss(y, y_hat)
        loss_history.append(loss)
        # 计算梯度
        dw, db = compute_gradient(X, y, y_hat)
        # 梯度下降更新参数
        w, b = gradient_descent_update(w, b, dw, db, lr)

        if epoch % 1000 == 0:
            print(f"迭代{epoch:5d} | 损失值: {loss:.4f}")
    return w, b, loss_history


# ===================== 3. 预测函数（输入4特征预测） =====================
def predict(X, w, b):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    pred = (y_hat > 0.5).astype(int)
    return pred, y_hat


# ===================== 4. 数据加载与预处理 =====================
# 加载iris数据集

X_raw, y = rd.ReadIris().binary_classification();

# 特征标准化（加速梯度下降收敛）
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_norm = (X_raw - X_mean) / X_std

# ===================== 5. 模型训练（4个特征完整训练） =====================
w_trained, b_trained, loss_hist = logistic_regression_train(X_norm, y)

# 训练集预测
y_pred, prob = predict(X_norm, w_trained, b_trained)
acc = np.mean(y_pred == y)
print(f"\n训练集准确率: {acc:.4f}")
print(f"训练权重w:\n{w_trained}")
print(f"偏置b: {b_trained:.4f}")

# ===================== 6. 绘图：选取2个特征可视化决策边界 =====================
# 选择前两个特征：sepal length, sepal width（绘图只用2维，训练仍用4维）
feat1_idx = 0
feat2_idx = 1
X_plot = X_norm[:, [feat1_idx, feat2_idx]]
x1_min, x1_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
x2_min, x2_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

# 生成网格点绘制分割线
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
grid = np.c_[xx1.ravel(), xx2.ravel()]

# 构造完整4维网格特征（补齐另外两个特征均值，保持输入维度一致）
grid_4d = np.zeros((grid.shape[0], 4))
grid_4d[:, feat1_idx] = grid[:, 0]
grid_4d[:, feat2_idx] = grid[:, 1]
# 剩余两特征填充标准化后的均值
for i in range(4):
    if i != feat1_idx and i != feat2_idx:
        grid_4d[:, i] = np.mean(X_norm[:, i])

# 网格预测
grid_pred, _ = predict(grid_4d, w_trained, b_trained)
grid_pred = grid_pred.reshape(xx1.shape)

# 画布绘制
plt.figure(figsize=(10, 6))
# 绘制分类底色分割区域
plt.contourf(xx1, xx2, grid_pred, alpha=0.3, cmap=plt.cm.coolwarm)
# 真实样本点
for label in [0, 1]:
    mask = (y.flatten() == label)
    c = "blue" if label == 0 else "red"
    label_name = "Setosa(0)" if label == 0 else "Versicolor(1)"
    plt.scatter(X_plot[mask, 0], X_plot[mask, 1], c=c, s=40, label=f"真实-{label_name}")

# 标记预测错误样本（黄色叉号）
wrong_mask = (y_pred.flatten() != y.flatten())
plt.scatter(X_plot[wrong_mask, 0], X_plot[wrong_mask, 1], c="yellow", marker="x", s=80, label="预测错误样本")

# 图表配置
plt.xlabel(f"特征1:  (标准化)")
plt.ylabel(f"特征2:  (标准化)")
plt.title("鸢尾花二分类逻辑回归 | 4特征训练，2特征可视化决策边界")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 绘制损失下降曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_hist)
plt.xlabel("迭代次数")
plt.ylabel("交叉熵损失")
plt.title("训练损失变化曲线")
plt.grid()
plt.show()


# ===================== 7. 自定义4特征输入预测示例 =====================
def predict_raw_input(raw_4feat):
    """输入原始4个花瓣特征，输出分类"""
    feat = np.array(raw_4feat).reshape(1, -1)
    feat_norm = (feat - X_mean) / X_std
    pred, prob = predict(feat_norm, w_trained, b_trained)
    cls = "Setosa(0)" if pred[0, 0] == 0 else "Versicolor(1)"
    return cls, prob[0, 0]


# 测试样例
test_sample1 = [5.1, 3.5, 1.4, 0.2]  # setosa
test_sample2 = [6.2, 2.9, 4.3, 1.3]  # versicolor
cls1, p1 = predict_raw_input(test_sample1)
cls2, p2 = predict_raw_input(test_sample2)
print(f"\n输入样本{test_sample1}")
print(f"预测类别: {cls1}, 正类概率: {p1:.4f}")
print(f"输入样本{test_sample2}")
print(f"预测类别: {cls2}, 正类概率: {p2:.4f}")
