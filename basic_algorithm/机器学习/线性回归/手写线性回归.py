from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import database.read_data as rd

# 1. 数据加载与预处理
def load_and_preprocess_data():
    # 加载加州房价数据集,指定字段
    field = ['median_income','median_house_value']
    housing = rd.CaliforniaHousing().get_assign_field(field)

    # 提取特征：MedInc (中位数收入) 和 Target (房价)
    # MedInc 是第0个特征
    x_raw = housing[:, 0] #.reshape(-1, 1)
    y_raw = housing[:, 1]

    return x_raw, y_raw


# 数据标准化
def standardization(data) -> Any:
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_norm = (data - data_mean) / data_std
    return data_norm, data_mean, data_std


# 2. 定义代价函数 (Mean Squared Error)
def compute_cost(pred, y):
    m = pred.shape[0]  # 样本数量
    cost = (1 / (2 * m)) * np.sum((pred - y) ** 2)
    return cost


# 3. 定义梯度下降算法
def gradient_descent(X, y, w_init, b_init, learning_rate, num_iterations):
    m = X.shape[0]
    w = w_init
    b = b_init
    cost_history = []

    for i in range(num_iterations):
        # 计算预测值
        predictions = w * X + b

        # 记录代价
        if i % 100 == 0:
            cost = compute_cost(predictions, y)
            cost_history.append(cost)

        # 计算梯度
        dw = (1 / m) * np.sum((predictions - y) * X)
        db = (1 / m) * np.sum(predictions - y)

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, cost_history


# 4. 主程序执行
def main():
    # 加载数据
    x_raw, y_raw = load_and_preprocess_data()

    # 标准化数据 (Feature Scaling) - 对梯度下降至关重要
    x_norm, x_mean, x_std = standardization(x_raw)
    y_norm, y_mean, y_std = standardization(y_raw)

    # 初始化参数
    w_init = 0
    b_init = 0
    learning_rate = 0.1  # 学习率
    num_iterations = 1000  # 迭代次数

    print("开始训练...")
    # 训练模型 (在标准化数据上训练)
    w_final, b_final, costs = gradient_descent(x_norm, y_norm, w_init, b_init, learning_rate, num_iterations)
    print(f"训练完成。最终权重 w: {w_final:.4f}, 偏置 b: {b_final:.4f}")

    # --- 预测三个特定的房价 ---
    # 假设我们要预测收入为 2.0, 5.0, 8.0 (原始单位: 万美元/年) 的房价
    test_incomes_raw = np.array([2.0, 5.0, 8.0]).reshape(-1, 1)

    # 先将测试数据标准化
    test_incomes_norm = (test_incomes_raw - x_mean) / x_std

    # 使用模型预测 (得到的是标准化后的房价)
    predicted_prices_norm = w_final * test_incomes_norm + b_final

    # 反标准化，得到原始单位的房价
    predicted_prices_raw = predicted_prices_norm * y_std + y_mean

    print("\n预测结果:")
    for i in range(len(test_incomes_raw)):
        print(f"收入: ${test_incomes_raw[i][0]:.0f} -> 预测房价: ${predicted_prices_raw[i][0]:.0f}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # 1. 绘制原始数据散点 (为了性能，只绘制部分数据点，避免过于密集)
    sample_indices = np.random.choice(x_raw.shape[0], 1000, replace=False)
    plt.scatter(x_raw[sample_indices], y_raw[sample_indices], color='blue', alpha=0.3, label='Sample Data Points')

    # 2. 绘制回归线
    # 生成一系列连续的X值用于画线
    x_line_raw = np.linspace(x_raw.min(), x_raw.max(), 100).reshape(-1, 1)
    x_line_norm = (x_line_raw - x_mean) / x_std
    y_line_norm = w_final * x_line_norm + b_final
    y_line_raw = y_line_norm * y_std + y_mean

    plt.plot(x_line_raw, y_line_raw, color='red', linewidth=2, label='Linear Regression Fit')

    # 3. 绘制三个预测点
    plt.scatter(test_incomes_raw, predicted_prices_raw, color='green', s=150, zorder=5,
                label='Predictions (2k, 5k, 8k Inc)')

    # 添加标注
    for i in range(len(test_incomes_raw)):
        plt.annotate(f'${predicted_prices_raw[i][0] * 100000:.0f}',
                     (test_incomes_raw[i][0], predicted_prices_raw[i][0]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.title('California Housing: Median Income vs Median House Value', fontsize=14)
    plt.xlabel('Median Income (in $10,000)', fontsize=12)
    plt.ylabel('Median House Value (in $100,000)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
