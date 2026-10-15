import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import database.read_data as rd

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_iris_data():
    """加载鸢尾花数据，取前100条做二分类"""
    try:
        X, y = rd.ReadIris().binary_classification()
    except FileNotFoundError:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data[:100]
        y = iris.target[:100]

    return X, y


def plot_decision_boundary(X, y, X_test, y_test, y_pred, model, scaler, feature_names):
    """绘制决策边界"""
    # 选择花瓣长度和花瓣宽度两个特征
    feature1_idx = 2  # 花瓣长度
    feature2_idx = 3  # 花瓣宽度

    # 创建网格
    x1_min, x1_max = X[:, feature1_idx].min() - 0.5, X[:, feature1_idx].max() + 0.5
    x2_min, x2_max = X[:, feature2_idx].min() - 0.5, X[:, feature2_idx].max() + 0.5
    h = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))

    # 构建完整的4维特征
    mean_vals = scaler.mean_
    scale_vals = scaler.scale_

    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    grid_full = np.zeros((grid_points.shape[0], 4))
    grid_full[:, feature1_idx] = grid_points[:, 0]
    grid_full[:, feature2_idx] = grid_points[:, 1]
    grid_full[:, 0] = mean_vals[0]
    grid_full[:, 1] = mean_vals[1]

    grid_scaled = (grid_full - mean_vals) / scale_vals

    # 预测
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx1.shape)

    # 绘图
    plt.figure(figsize=(10, 8))

    # 绘制决策边界
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')

    # 绘制训练数据
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:, feature1_idx], X[:, feature2_idx],
                c=colors, s=80, alpha=0.6, edgecolors='black',
                linewidth=1.5, label='训练数据')

    # 绘制测试数据
    X_test_orig = X_test * scale_vals + mean_vals
    for i in range(len(X_test_orig)):
        if y_pred[i] == 0:
            color = 'green'
            marker = '^'  # 三角形
        else:
            color = 'orange'
            marker = 's'  # 正方形

        # 预测正确用黑色边框，错误用红色边框
        edgecolor = 'black' if y_pred[i] == y_test[i] else 'red'
        plt.scatter(X_test_orig[i, feature1_idx], X_test_orig[i, feature2_idx],
                    c=color, s=150, marker=marker,
                    edgecolors=edgecolor, linewidth=2, alpha=0.9)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Setosa (训练)'),
        Patch(facecolor='red', alpha=0.6, label='Versicolour (训练)'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                   markersize=10, label='预测: Setosa'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
                   markersize=10, label='预测: Versicolour'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='red', markersize=10, label='预测错误')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.xlabel('花瓣长度', fontsize=12)
    plt.ylabel('花瓣宽度', fontsize=12)
    plt.title('逻辑回归 - 鸢尾花分类决策边界', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("使用 sklearn 实现逻辑回归 - 鸢尾花二分类")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    X, y = load_iris_data()
    print(f"  数据形状: X={X.shape}, y={y.shape}")
    print(f"  类别分布: Setosa={np.sum(y == 0)}, Versicolour={np.sum(y == 1)}")

    # 2. 数据预处理
    print("\n2. 数据预处理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {X_train.shape[0]}个样本")
    print(f"  测试集: {X_test.shape[0]}个样本")

    # 4. 训练模型
    print("\n3. 训练逻辑回归模型...")
    model = LogisticRegression(
        penalty='l2',  # L2正则化
        max_iter=1000,  # 最大迭代次数
        random_state=42
    )
    model.fit(X_train, y_train)

    # 打印模型参数
    print(f"\n  模型参数:")
    print(f"    偏置 (intercept): {model.intercept_[0]:.6f}")
    feature_names = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']
    for i, name in enumerate(feature_names):
        print(f"    权重 {name}: {model.coef_[0][i]:.6f}")

    # 5. 预测
    print("\n4. 模型预测...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 6. 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  测试集准确率: {accuracy:.4f}")

    # 打印预测结果
    print("\n  预测结果示例（前5个）:")
    for i in range(min(5, len(X_test))):
        species = 'Setosa' if y_pred[i] == 0 else 'Versicolour'
        prob = y_proba[i][1]  # 预测为类别1的概率
        print(f"    真实: {y_test[i]}, 预测: {y_pred[i]} ({species}), 概率: {prob:.4f}")

    # 7. 可视化
    print("\n5. 可视化决策边界...")
    plot_decision_boundary(X, y, X_test, y_test, y_pred, model, scaler, feature_names)

    # 8. 用户输入预测
    print("\n" + "=" * 60)
    print("6. 用户输入预测")
    print("请输入4个特征值 (萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度):")
    print("示例: 5.1 3.5 1.4 0.2")

    try:
        user_input = input("\n请输入4个数值，用空格分隔: ").strip().split()
        if len(user_input) == 4:
            features = np.array([float(x) for x in user_input]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)
            species = 'Setosa' if pred[0] == 0 else 'Versicolour'
            print(f"\n预测结果: {species}")
            print(f"概率: Setosa={proba[0][0]:.4f}, Versicolour={proba[0][1]:.4f}")

            # 重新绘制决策边界并标记用户输入
            feature1_idx = 2
            feature2_idx = 3
            mean_vals = scaler.mean_
            scale_vals = scaler.scale_

            plt.figure(figsize=(8, 6))
            # 绘制决策边界
            x1_min, x1_max = X[:, feature1_idx].min() - 0.5, X[:, feature1_idx].max() + 0.5
            x2_min, x2_max = X[:, feature2_idx].min() - 0.5, X[:, feature2_idx].max() + 0.5
            h = 0.02
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
            grid_points = np.c_[xx1.ravel(), xx2.ravel()]
            grid_full = np.zeros((grid_points.shape[0], 4))
            grid_full[:, feature1_idx] = grid_points[:, 0]
            grid_full[:, feature2_idx] = grid_points[:, 1]
            grid_full[:, 0] = mean_vals[0]
            grid_full[:, 1] = mean_vals[1]
            grid_scaled = (grid_full - mean_vals) / scale_vals
            Z = model.predict(grid_scaled)
            Z = Z.reshape(xx1.shape)
            plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')

            # 绘制训练数据
            colors = ['blue' if label == 0 else 'red' for label in y]
            plt.scatter(X[:, feature1_idx], X[:, feature2_idx],
                        c=colors, s=50, alpha=0.5, edgecolors='black')

            # 标记用户输入
            plt.scatter(features[0, feature1_idx], features[0, feature2_idx],
                        c='yellow', s=300, marker='*', edgecolors='black',
                        linewidth=2, label=f'输入点: {species}')

            plt.xlabel('花瓣长度')
            plt.ylabel('花瓣宽度')
            plt.title(f'预测结果: {species}', fontweight='bold')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.2)
            plt.show()
        else:
            print("请输入4个数值！")
    except ValueError as e:
        print(f"输入格式错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()