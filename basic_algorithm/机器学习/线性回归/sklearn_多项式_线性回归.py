import numpy as np
import matplotlib.pyplot as plt
import database.read_data as rd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# 1. 数据加载
def load_data():
    field = ['median_income','median_house_value']
    housing = rd.CaliforniaHousing().get_assign_field(field)
    x_raw = housing[:, 0].reshape(-1, 1)
    y_raw = housing[:, 1]
    return x_raw, y_raw

# 2. 构建多项式回归管道
def build_polynomial_pipeline(degree=2):
    """
    创建一个包含多项式特征生成、标准化和线性回归的管道
    """
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        # 生成多项式特征，不包含bias(因为后面有LinearRegression会处理intercept)
        ('scaler', StandardScaler()),  # 标准化多项式特征
        ('lin_reg', LinearRegression())  # 线性回归模型
    ])
    return pipeline


# 3. 主程序
def main():
    X_raw, y_raw = load_data()

    # --- 训练模型 ---
    # 我们尝试 2阶 多项式
    degree = 2
    poly_pipeline = build_polynomial_pipeline(degree=degree)

    print(f"正在训练 {degree} 阶多项式回归模型...")
    poly_pipeline.fit(X_raw, y_raw)

    # 查看模型得分 (R^2)
    score = poly_pipeline.score(X_raw, y_raw)
    print(f"训练集 R^2 分数: {score:.4f}")

    # --- 预测三个特定房价 ---
    test_incomes_raw = np.array([2.0, 5.0, 8.0]).reshape(-1, 1)

    # 直接使用 pipeline 预测，内部会自动处理多项式转换和标准化
    predicted_prices_raw = poly_pipeline.predict(test_incomes_raw)

    print("\n多项式回归预测结果 (Sklearn Pipeline):")
    for i in range(len(test_incomes_raw)):
        print(f"收入: ${test_incomes_raw[i][0] * 10000:.0f} -> 预测房价: ${predicted_prices_raw[i] * 100000:.0f}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # 1. 绘制散点
    sample_indices = np.random.choice(X_raw.shape[0], 1000, replace=False)
    plt.scatter(X_raw[sample_indices], y_raw[sample_indices], color='blue', alpha=0.3, label='Sample Data Points')

    # 2. 绘制多项式回归曲线
    # 生成密集的X点用于画平滑曲线
    x_line_raw = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)
    y_line_raw = poly_pipeline.predict(x_line_raw)

    plt.plot(x_line_raw, y_line_raw, color='red', linewidth=2.5, label=f'Polynomial Fit (Degree {degree})')

    # 3. 绘制预测点
    plt.scatter(test_incomes_raw, predicted_prices_raw, color='green', s=150, zorder=5,
                label='Predictions (2k, 5k, 8k Inc)')

    # 添加标注
    for i in range(len(test_incomes_raw)):
        plt.annotate(f'${predicted_prices_raw[i] * 100000:.0f}',
                     (test_incomes_raw[i][0], predicted_prices_raw[i]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9)

    plt.title(f'California Housing: Polynomial Regression (Degree {degree}) via Sklearn Pipeline', fontsize=14)
    plt.xlabel('Median Income (in $10,000)', fontsize=12)
    plt.ylabel('Median House Value (in $100,000)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # --- 额外演示：对比不同阶数的效果 ---
    print("\n--- 不同阶数多项式对比 ---")
    for d in [1, 2, 3, 5]:
        p = build_polynomial_pipeline(degree=d)
        p.fit(X_raw, y_raw)
        print(f"Degree {d}: R^2 = {p.score(X_raw, y_raw):.4f}")


if __name__ == "__main__":
    main()