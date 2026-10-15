import database.read_data as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# 1. 数据加载
def load_data():
    field = ['median_income','median_house_value']
    housing = rd.CaliforniaHousing().get_assign_field(field)
    x_raw = housing[:, 0].reshape(-1, 1)
    y_raw = housing[:, 1]
    return x_raw, y_raw


# 2. 主程序
def main():
    # 加载原始数据
    X_raw, y_raw = load_data()

    # --- 数据预处理 (使用 Sklearn StandardScaler) ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # fit_transform 同时计算均值/方差并转换数据
    X_norm = scaler_X.fit_transform(X_raw)
    y_norm = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()  # ravel 将二维转为一维数组

    # --- 模型训练 ---
    model = LinearRegression()
    model.fit(X_norm, y_norm)

    # 获取训练好的参数 (注意：这是在标准化空间下的参数)
    w_sklearn = model.coef_[0]
    b_sklearn = model.intercept_

    print(f"Sklearn Model - Weight (w): {w_sklearn:.4f}, Bias (b): {b_sklearn:.4f}")

    # --- 预测三个特定房价 ---
    test_incomes_raw = np.array([2.0, 5.0, 8.0]).reshape(-1, 1)

    # 重要：预测前必须先对输入数据进行相同的标准化处理
    test_incomes_norm = scaler_X.transform(test_incomes_raw)

    # 模型预测 (得到标准化后的房价)
    predicted_prices_norm = model.predict(test_incomes_norm)

    # 反标准化：将预测结果还原回原始单位
    # inverse_transform 需要二维数组，所以 reshape(-1, 1)
    predicted_prices_raw = scaler_y.inverse_transform(predicted_prices_norm.reshape(-1, 1)).ravel()

    print("\n预测结果 (Sklearn):")
    for i in range(len(test_incomes_raw)):
        print(f"收入: ${test_incomes_raw[i][0]:.0f} -> 预测房价: ${predicted_prices_raw[i]:.0f}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # 1. 绘制散点 (随机采样1000个点以保持清晰)
    sample_indices = np.random.choice(X_raw.shape[0], 1000, replace=False)
    plt.scatter(X_raw[sample_indices], y_raw[sample_indices], color='blue', alpha=0.3, label='Sample Data Points')

    # 2. 绘制回归线
    # 生成用于画线的X轴数据
    x_line_raw = np.linspace(X_raw.min(), X_raw.max(), 100).reshape(-1, 1)
    # 标准化
    x_line_norm = scaler_X.transform(x_line_raw)
    # 预测标准化Y
    y_line_norm = model.predict(x_line_norm)
    # 反标准化Y
    y_line_raw = scaler_y.inverse_transform(y_line_norm.reshape(-1, 1)).ravel()

    plt.plot(x_line_raw, y_line_raw, color='red', linewidth=2, label='Linear Regression Fit (Sklearn)')

    # 3. 绘制预测点
    plt.scatter(test_incomes_raw, predicted_prices_raw, color='green', s=150, zorder=5,
                label='Predictions (2k, 5k, 8k Inc)')

    # 添加标注
    for i in range(len(test_incomes_raw)):
        plt.annotate(f'${predicted_prices_raw[i] * 100000:.0f}',
                     (test_incomes_raw[i][0], predicted_prices_raw[i]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.title('California Housing: Median Income vs Median House Value (Sklearn)', fontsize=14)
    plt.xlabel('Median Income (in $10,000)', fontsize=12)
    plt.ylabel('Median House Value (in $100,000)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# fit_transform 与 transform 的区别：
# 在训练集上，我们使用 scaler_X.fit_transform(X_raw)，这会计算训练集的均值和方差，并同时进行转换。
# 在测试集（那三个预测点）或画线数据上，我们必须使用 scaler_X.transform(...)(带入计算fit时候计算过的均值和方差)
# 绝对不能再次重新 fit，否则会使用测试数据的统计量(用测试数据重新计算均值和方差)，导致数据分布不一致，预测结果错误。

# StandardScaler 的 inverse_transform
# 方法通常期望输入是二维数组 (n_samples, n_features)。
# 因此，在反标准化预测结果时，我使用了 .reshape(-1, 1)，然后再用 .ravel() 变回一维以便绘图。