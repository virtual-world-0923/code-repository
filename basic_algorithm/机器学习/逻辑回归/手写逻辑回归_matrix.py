import numpy as np
import matplotlib.pyplot as plt
import warnings
import database.read_data as rd
warnings.filterwarnings("ignore")

# ================= 1. 读取数据与预处理 =================
# 读取 iris.data 文件
X_raw ,y = rd.ReadIris().binary_classification()

# 特征标准化（逻辑回归强烈建议标准化，以加速收敛）
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_scaled = (X_raw - X_mean) / X_std

# 【核心改动】在特征矩阵前面添加一列 1，用于合并偏置项 b
m, n_features = X_scaled.shape
X_with_bias = np.hstack([np.ones((m, 1)), X_scaled])  # 形状变为 (100, 5)


# ================= 2. 单独激活函数方法 =================
def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ================= 3. 单独代价损失方法 =================
def compute_cost(X, y, theta):
    """计算二元交叉熵损失"""
    h = sigmoid(X.dot(theta))
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


# ================= 4. 矩阵方式梯度下降方法 =================
def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000):
    """
    矩阵化梯度下降
    theta 包含了 b 和所有的 w，一次性全部更新
    """
    cost_history = []
    for i in range(iterations):
        # 前向传播
        h = sigmoid(X.dot(theta))

        # 误差
        error = h - y  # shape: (100, 1)

        # 【核心改动】矩阵方式计算梯度并更新
        # X.T 形状: (5, 100), error 形状: (100, 1)
        # 结果 grad 形状: (5, 1)，同时包含了 db 和 dw1~dw4
        grad = (1 / m) * X.T.dot(error)

        # 一次性更新所有参数 (b 和 w)
        theta = theta - learning_rate * grad

        # 记录代价
        if i % 200 == 0:
            cost = compute_cost(X, y, theta)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return theta, cost_history


# ================= 5. 训练数据 =================
print("开始矩阵化逻辑回归训练...")
# 初始化参数为 0，形状为 (5, 1)
theta_init = np.zeros((n_features + 1, 1))

theta_final, costs = gradient_descent(X_with_bias, y, theta_init)
print("训练完成！\n")


# ================= 6. 预测函数 =================
def predict_flower(features_raw, theta, X_mean, X_std):
    """输入原始4个特征，预测花的种类"""
    # 标准化
    features_scaled = (features_raw - X_mean) / X_std
    # 添加 bias 列
    bias_column = np.ones((features_scaled.shape[0], 1))
    features_with_bias = np.hstack([bias_column, features_scaled])

    prob = sigmoid(features_with_bias.dot(theta))[0][0]
    return "Iris-setosa" if prob < 0.5 else "Iris-versicolor"


# 测试预测
test_sample_1 = np.array([[5.1, 3.5, 1.4, 0.2]])  # 典型的 Setosa
test_sample_2 = np.array([[6.0, 2.7, 4.5, 1.5]])  # 典型的 Versicolor

print(f"样本1预测结果: {predict_flower(test_sample_1, theta_final, X_mean, X_std)}")
print(f"样本2预测结果: {predict_flower(test_sample_2, theta_final, X_mean, X_std)}\n")

# ================= 7. 简化版画图 =================
# 选择 Petal Length (索引2) 和 Petal Width (索引3) 作为绘图特征
feat_x_idx, feat_y_idx = 2, 3

plt.figure(figsize=(9, 6))

# 1. 画出训练数据：两种不同类别用不同颜色
plt.scatter(X_scaled[y.ravel() == 0, feat_x_idx], X_scaled[y.ravel() == 0, feat_y_idx],
            color='blue', edgecolor='k', label='True: Iris-setosa', s=60)
plt.scatter(X_scaled[y.ravel() == 1, feat_x_idx], X_scaled[y.ravel() == 1, feat_y_idx],
            color='red', edgecolor='k', label='True: Iris-versicolor', s=60)

# 2. 画出预测数据：用不同的形状（如五角星）和颜色（如绿色）
# 这里用训练集上的预测结果做演示
ts1 = features_scaled = (test_sample_1 - X_mean) / X_std
ts2 = features_scaled = (test_sample_2 - X_mean) / X_std
test_sample = np.vstack([ts1,ts2])
bias_column = np.ones((test_sample.shape[0], 1))
features_with_bias = np.hstack([bias_column, test_sample])
pred_probs = sigmoid(features_with_bias.dot(theta_final))

pred_labels = (pred_probs > 0.5).astype(int).ravel()

plt.scatter(test_sample[pred_labels == 0, feat_x_idx], test_sample[pred_labels == 0, feat_y_idx],
            marker='*', color='green', s=150, label='Pred: Iris-setosa')
plt.scatter(test_sample[pred_labels == 1, feat_x_idx], test_sample[pred_labels == 1, feat_y_idx],
            marker='*', color='orange', s=150, label='Pred: Iris-versicolor')

plt.title('Simplified Logistic Regression Visualization', fontsize=14)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()