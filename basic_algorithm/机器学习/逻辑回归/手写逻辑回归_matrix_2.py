import numpy as np
import matplotlib.pyplot as plt
import database.read_data as rd


# ============================================================
# 2、激活函数（单独方法）
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 3、代价损失函数（单独方法）
# ============================================================
def compute_loss(y, y_pred):
    m = len(y)
    return -1 / m * np.sum(y * np.log(y_pred + 1e-8) +(1 - y) * np.log(1 - y_pred + 1e-8)
    )


# ============================================================
# 4、梯度计算方法（单独方法）
# ============================================================
def compute_gradient(X, y, y_pred):
    m = len(y)
    return 1 / m * X.T @ (y_pred - y)


# ============================================================
# 4、梯度下降（矩阵方式整体更新参数，含偏置）
# ============================================================
def gradient_descent(X, y, lr=0.1, epochs=2000):
    m, n = X.shape
    theta = np.zeros(n).reshape(-1,1)  # 参数矩阵（第0位就是偏置b）

    for i in range(epochs):
        z = X @ theta
        y_pred = sigmoid(z)

        gradient = compute_gradient(X, y, y_pred)
        theta -= lr * gradient

        if i % 200 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {i:4d}, Loss: {loss:.4f}")

    return theta


# ============================================================
# 6、预测函数（输入4个特征）
# ============================================================
def predict(X_raw, theta):
    """X_raw: (n_samples, 4) 原始4特征"""
    X_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])
    prob = sigmoid(X_bias @ theta)
    return (prob >= 0.5).astype(int), prob


# ============================================================
# 1、读取数据（前100条，二分类）
# ============================================================


X_raw ,y = rd.ReadIris().binary_classification()

# 在 X 前加一列 1，统一表示偏置 b
X_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])


# ============================================================
# 5、训练模型（4个特征）
# ============================================================
theta = gradient_descent(X_bias, y, lr=0.1, epochs=2000)

print("\n训练完成!")
print(f"参数 theta (b, w1, w2, w3, w4): {np.round(theta, 4)}")


# ============================================================
# 6、示例预测（输入4个特征）
# ============================================================
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred_class, pred_prob = predict(sample, theta)
print(f"\n预测类别: {pred_class[0]}  (概率: {pred_prob[0]})")


# ============================================================
# 7、取两个特征画二维图（训练用4维，画图只用2维）
# ============================================================
feature1 = X_raw[:, 0].reshape(-1,1)  # 萼片长度
feature2 = X_raw[:, 1].reshape(-1,1)  # 萼片宽度

class_0 = y == 0
class_1 = y == 1

plt.figure(figsize=(7, 5))

# 训练数据 —— 两类不同颜色
plt.scatter(feature1[class_0], feature2[class_0],color='red', label='Setosa (Class 0)', alpha=0.7)
plt.scatter(feature1[class_1], feature2[class_1],color='blue', label='Versicolor (Class 1)', alpha=0.7)

# 分割线：theta0 + theta1*x1 + theta2*x2 = 0
# 这里只取前两个特征的权重画二维分割线
x1_line = np.linspace(feature1.min() - 0.5, feature1.max() + 0.5, 100)
x2_line = -(theta[0] + theta[1] * x1_line) / theta[2]
plt.plot(x1_line, x2_line, 'k--', linewidth=2, label='Decision Boundary')

# 预测样本 —— 不同形状 + 颜色
# 取几个测试样本
test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.5, 3.0, 5.5, 1.8],
    [5.8, 2.6, 4.0, 1.2]
])

for ts in test_samples:
    ts_2d = ts[:2].reshape(1, -1)
    p_class, p_prob = predict(ts.reshape(1, -1), theta)

    marker = 'X' if p_class[0] == 0 else 'P'
    color = 'lime' if p_class[0] == 0 else 'gold'

    plt.scatter(ts_2d[:, 0], ts_2d[:, 1],
                color=color, marker=marker, s=250,
                edgecolors='black', linewidths=1.5,
                label=f'Pred={p_class[0]} (p={p_prob[0]})')

# 去重 legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

plt.xlabel('Sepal Length (Feature 1)')
plt.ylabel('Sepal Width (Feature 2)')
plt.title('Logistic Regression — Training Data + Predictions')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()