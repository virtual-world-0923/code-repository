import numpy as np
import matplotlib.pyplot as plt
import database.read_data as rd


# 1. 数据加载
def load_data():
    # 加载加州房价数据集,指定字段
    field = ['median_income', 'median_house_value']
    housing = rd.CaliforniaHousing().get_assign_field(field)

    # 提取特征：MedInc (中位数收入) 和 Target (房价)
    # MedInc 是第0个特征
    x_raw = housing[:, 0] .reshape(-1, 1)
    y_raw = housing[:, 1]
    return x_raw, y_raw


# 2. 构建多项式特征 (手动实现 PolyFeatures)
def create_polynomial_features(X, degree=2):
    """
    生成多项式特征: [1, x, x^2, ..., x^degree]
    """
    m = X.shape[0]
    # 初始化全1列 (Bias)
    X_poly = np.ones((m, 1))

    for d in range(1, degree + 1):
        # 计算 X 的 d 次方
        col = X ** d
        # 拼接到矩阵右侧
        X_poly = np.hstack([X_poly, col])

    return X_poly


# 3. 矩阵化梯度下降 (通用版，支持任意维度)
def gradient_descent_matrix(X_poly, y, theta_init, learning_rate, num_iterations):
    m = X_poly.shape[0]
    theta = theta_init.copy()

    for i in range(num_iterations):
        predictions = X_poly.dot(theta)
        error = predictions - y

        # 梯度: (1/m) * X^T * error
        gradient = (1 / m) * X_poly.T.dot(error)

        # 更新
        theta = theta - learning_rate * gradient

    return theta


# 4. 主程序
def main():
    X_raw, y_raw = load_data()

    # --- 关键步骤：标准化原始特征 ---
    # 注意：多项式回归必须在标准化后的特征上计算幂次，或者计算幂次后再标准化
    # 这里我们采用：先标准化 X，再生成多项式，再对多项式特征整体标准化

    X_mean = np.mean(X_raw)
    X_std = np.std(X_raw)
    X_norm = (X_raw - X_mean) / X_std

    y_mean = np.mean(y_raw)
    y_std = np.std(y_raw)
    y_norm = (y_raw - y_mean) / y_std

    # --- 生成多项式特征 (2阶: 1, x, x^2) ---
    # X_poly_norm 形状: (m, 3) -> [1, x_norm, x_norm^2]
    X_poly_norm = create_polynomial_features(X_norm, degree=5)

    # 对多项式特征再次标准化 (除了第一列 Bias 1)
    # 为什么？因为 x^2 的方差和 x 不同，量级差异会导致梯度下降困难
    poly_scaler_mean = np.mean(X_poly_norm[:, 1:], axis=0)  # 排除 bias 列
    poly_scaler_std = np.std(X_poly_norm[:, 1:], axis=0)

    # 手动标准化第1列和第2列 (索引1和2)
    X_poly_norm[:, 1:] = (X_poly_norm[:, 1:] - poly_scaler_mean) / poly_scaler_std

    y_vec = y_norm.reshape(-1, 1)

    # 初始化参数 (3个参数: b, w1, w2)
    theta_init = np.zeros((6, 1))
    learning_rate = 0.1
    num_iterations = 2000

    print("开始多项式回归训练 (Degree=2)...")
    theta_final = gradient_descent_matrix(X_poly_norm, y_vec, theta_init, learning_rate, num_iterations)

    print(f"参数 Theta: {theta_final.flatten()}")

    # --- 预测三个点 ---
    test_incomes_raw = np.array([2.0, 5.0, 8.0]).reshape(-1, 1)
    test_incomes_norm = (test_incomes_raw - X_mean) / X_std

    # 生成测试集的多项式特征
    test_poly_norm = create_polynomial_features(test_incomes_norm, degree=5)
    # 应用相同的标准化
    test_poly_norm[:, 1:] = (test_poly_norm[:, 1:] - poly_scaler_mean) / poly_scaler_std

    # 预测
    pred_norm = test_poly_norm.dot(theta_final)
    pred_raw = pred_norm * y_std + y_mean

    print("\n多项式回归预测结果:")
    for i in range(len(test_incomes_raw)):
        print(f"收入: ${test_incomes_raw[i][0] * 10000:.0f} -> 预测房价: ${pred_raw[i][0] * 100000:.0f}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # 1. 散点
    sample_indices = np.random.choice(X_raw.shape[0], 1000, replace=False)
    plt.scatter(X_raw[sample_indices], y_raw[sample_indices], color='blue', alpha=0.3, label='Data Points')

    # 2. 绘制多项式曲线
    x_line_raw = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)
    x_line_norm = (x_line_raw - X_mean) / X_std

    # 生成线条的多项式特征并标准化
    x_line_poly = create_polynomial_features(x_line_norm, degree=5)
    x_line_poly[:, 1:] = (x_line_poly[:, 1:] - poly_scaler_mean) / poly_scaler_std

    # 预测 Y
    y_line_norm = x_line_poly.dot(theta_final)
    y_line_raw = y_line_norm * y_std + y_mean

    plt.plot(x_line_raw, y_line_raw, color='red', linewidth=2.5, label='Polynomial Fit (Degree 2)')

    # 3. 预测点
    plt.scatter(test_incomes_raw, pred_raw, color='green', s=150, zorder=5, label='Predictions')

    for i in range(len(test_incomes_raw)):
        plt.annotate(f'${pred_raw[i][0] * 100000:.0f}',
                     (test_incomes_raw[i][0], pred_raw[i][0]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.title('California Housing: Polynomial Regression (MedInc)', fontsize=14)
    plt.xlabel('Median Income (in $10,000)', fontsize=12)
    plt.ylabel('Median House Value (in $100,000)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()