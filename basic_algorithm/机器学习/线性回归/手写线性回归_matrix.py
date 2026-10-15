import database.read_data as rd
import numpy as np
import matplotlib.pyplot as plt


# 1. 数据加载与预处理
def load_and_preprocess_data():
    field = ['median_income', 'median_house_value']
    housing = rd.CaliforniaHousing().get_assign_field(field)

    # 提取特征：MedInc (中位数收入) 和 Target (房价)
    # MedInc 是第0个特征
    x_raw = housing[:, 0].reshape(-1, 1)
    y_raw = housing[:, 1]

    # 标准化
    X_mean = np.mean(x_raw)
    X_std = np.std(x_raw)
    X_norm = (x_raw - X_mean) / X_std

    y_mean = np.mean(y_raw)
    y_std = np.std(y_raw)
    y_norm = (y_raw - y_mean) / y_std

    return x_raw, y_raw, X_norm, y_norm, X_mean, X_std, y_mean, y_std


# 2. 添加 Bias 列 (全1列)
def add_bias_column(X):
    """
    在矩阵 X 的最左侧添加一列全为 1 的向量
    输入: X (m, 1)
    输出: X_b (m, 2) -> [1, x]
    """
    m = X.shape[0]
    # np.ones((m, 1)) 创建 m行1列的全1矩阵
    bias_col = np.ones((m, 1))
    # hstack 水平拼接
    X_b = np.hstack([bias_col, X])
    return X_b


# 3. 矩阵形式的梯度下降
def gradient_descent_matrix(X_b, y, theta_init, learning_rate, num_iterations):
    """
    X_b: 已添加 Bias 列的特征矩阵 (m, 2)
    y: 目标值向量 (m, 1)
    theta_init: 初始参数向量 [b, w] (2, 1)
    """
    m = X_b.shape[0]
    theta = theta_init.copy()
    cost_history = []

    for i in range(num_iterations):
        # 1. 预测: h = X * theta
        # X_b (m, 2) @ theta (2, 1) -> predictions (m, 1)
        predictions = X_b.dot(theta)

        # 2. 计算误差
        error = predictions - y

        # 3. 计算代价 (可选，用于监控收敛)
        if i % 100 == 0:
            cost = (1 / (2 * m)) * np.sum(np.square(error))
            cost_history.append(cost)

        # 4. 计算梯度 (矩阵形式)
        # gradient = (1/m) * X^T * error
        # X_b.T (2, m) @ error (m, 1) -> gradient (2, 1)
        gradient = (1 / m) * X_b.T.dot(error)

        # 5. 更新参数
        theta = theta - learning_rate * gradient

    return theta, cost_history


# 4. 主程序
def main():
    # 加载数据
    X_raw, y_raw, X_norm, y_norm, X_mean, X_std, y_mean, y_std = load_and_preprocess_data()

    # --- 关键步骤：添加 Bias 列 ---
    X_b = add_bias_column(X_norm)

    # 将 y 转换为列向量 (m, 1) 以匹配矩阵运算
    y_vec = y_norm.reshape(-1, 1)

    # 初始化参数 theta = [b, w]^T
    theta_init = np.zeros((2, 1))
    learning_rate = 0.1
    num_iterations = 1000

    print("开始矩阵化梯度下降训练...")
    theta_final, costs = gradient_descent_matrix(X_b, y_vec, theta_init, learning_rate, num_iterations)

    b_final = theta_final[0][0]
    w_final = theta_final[1][0]

    print(f"训练完成。")
    print(f"偏置 b (Theta[0]): {b_final:.4f}")
    print(f"权重 w (Theta[1]): {w_final:.4f}")

    # --- 预测三个特定房价 ---
    test_incomes_raw = np.array([2.0, 5.0, 8.0]).reshape(-1, 1)

    # 1. 标准化测试数据
    test_incomes_norm = (test_incomes_raw - X_mean) / X_std

    # 2. 添加 Bias 列
    test_incomes_b = add_bias_column(test_incomes_norm)

    # 3. 矩阵预测
    predicted_prices_norm = test_incomes_b.dot(theta_final)

    # 4. 反标准化
    predicted_prices_raw = predicted_prices_norm * y_std + y_mean

    print("\n预测结果:")
    for i in range(len(test_incomes_raw)):
        print(f"收入: ${test_incomes_raw[i][0]:.0f} -> 预测房价: ${predicted_prices_raw[i][0] :.0f}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # 1. 散点图
    sample_indices = np.random.choice(X_raw.shape[0], 1000, replace=False)
    plt.scatter(X_raw[sample_indices], y_raw[sample_indices], color='blue', alpha=0.3, label='Sample Data Points')

    # 2. 回归线
    x_line_raw = np.linspace(X_raw.min(), X_raw.max(), 100).reshape(-1, 1)
    x_line_norm = (x_line_raw - X_mean) / X_std
    x_line_b = add_bias_column(x_line_norm)

    # 矩阵预测线条 Y 值
    y_line_norm = x_line_b.dot(theta_final)
    y_line_raw = y_line_norm * y_std + y_mean

    plt.plot(x_line_raw, y_line_raw, color='red', linewidth=2, label='Linear Regression Fit (Matrix)')

    # 3. 预测点
    plt.scatter(test_incomes_raw, predicted_prices_raw, color='green', s=150, zorder=5,
                label='Predictions (2k, 5k, 8k Inc)')

    # 标注
    for i in range(len(test_incomes_raw)):
        plt.annotate(f'${predicted_prices_raw[i][0] * 100000:.0f}',
                     (test_incomes_raw[i][0], predicted_prices_raw[i][0]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.title('California Housing: Matrix Linear Regression', fontsize=14)
    plt.xlabel('Median Income (in $10,000)', fontsize=12)
    plt.ylabel('Median House Value (in $100,000)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()