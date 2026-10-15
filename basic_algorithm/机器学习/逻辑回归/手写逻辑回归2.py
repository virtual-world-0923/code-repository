import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """激活函数 - Sigmoid"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """代价损失函数 - 二分类交叉熵"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_gradient(self, X, y_true, y_pred):
        """计算梯度"""
        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)
        return dw, db

    def gradient_descent_step(self, X, y_true, y_pred):
        """梯度下降更新参数"""
        dw, db = self.compute_gradient(X, y_true, y_pred)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return dw, db

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            self.gradient_descent_step(X, y, y_pred)

            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss:.4f}')

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def load_iris_data():
    """加载鸢尾花数据，取前100条做二分类"""
    try:
        data = pd.read_csv('iris.data', header=None)
    except FileNotFoundError:
        # 如果文件不存在，从sklearn加载
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(np.c_[iris.data, iris.target_names[iris.target]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

    # 取前100条数据（Setosa和Versicolour）
    data_100 = data.iloc[:100, :]

    # 特征：4个特征
    X = data_100.iloc[:, :4].values.astype(float)

    # 标签：将Setosa设为0，Versicolour设为1
    y = data_100.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    return X, y


def main():
    # 1. 加载数据
    print("=" * 50)
    print("1. 加载鸢尾花数据集（前100条）")
    X, y = load_iris_data()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: 类别0 (Setosa)={np.sum(y == 0)}, 类别1 (Versicolour)={np.sum(y == 1)}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {X_train.shape[0]}个样本, 测试集: {X_test.shape[0]}个样本")

    # 2. 创建并训练模型
    print("\n" + "=" * 50)
    print("2. 训练逻辑回归模型")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # 3. 预测
    print("\n" + "=" * 50)
    print("3. 模型预测")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    # 打印预测结果
    print("\n预测结果示例（前5个）:")
    for i in range(5):
        species = 'Setosa' if y_pred[i] == 0 else 'Versicolour'
        # y_proba[i] 是标量，直接使用
        prob = y_proba[i]  # 这是预测为类别1的概率
        print(f"  真实值: {y_test[i]}, 预测值: {y_pred[i]} ({species}), 概率(类别1): {prob:.4f}")

    # 4. 可视化损失曲线
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, linewidth=2)
    plt.title('Training Loss Curve', fontsize=12)
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Loss Value', fontsize=11)
    plt.grid(True, alpha=0.3)

    # 5. 使用两个特征进行可视化
    print("\n" + "=" * 50)
    print("4. 可视化决策边界")

    # 使用原始特征（未标准化）进行可视化
    X_orig = X
    feature1_idx = 2  # 特征3 (petal length)
    feature2_idx = 3  # 特征4 (petal width)

    # 创建网格
    x1_min, x1_max = X_orig[:, feature1_idx].min() - 0.5, X_orig[:, feature1_idx].max() + 0.5
    x2_min, x2_max = X_orig[:, feature2_idx].min() - 0.5, X_orig[:, feature2_idx].max() + 0.5

    h = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))

    # 构建完整的4维特征向量
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]

    mean_vals = scaler.mean_
    scale_vals = scaler.scale_

    grid_full = np.zeros((grid_points.shape[0], 4))
    grid_full[:, feature1_idx] = grid_points[:, 0]
    grid_full[:, feature2_idx] = grid_points[:, 1]
    # 对其他特征使用均值
    other_features = [0, 1]
    for idx in other_features:
        grid_full[:, idx] = mean_vals[idx]

    # 标准化网格点
    grid_scaled = (grid_full - mean_vals) / scale_vals

    # 预测
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx1.shape)

    # 绘图
    plt.subplot(1, 2, 2)

    # 绘制决策边界
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm', levels=[-0.1, 0.5, 1.1])

    # 绘制所有数据点
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X_orig[:, feature1_idx], X_orig[:, feature2_idx],
                c=colors, s=50, edgecolors='black', alpha=0.7, label='Training Data')

    # 绘制测试集预测结果
    X_test_orig = X_test * scale_vals + mean_vals
    for i in range(len(X_test_orig)):
        marker = 'o' if y_pred[i] == y_test[i] else 'x'
        color = 'lime' if y_pred[i] == 0 else 'orange'
        plt.scatter(X_test_orig[i, feature1_idx], X_test_orig[i, feature2_idx],
                    c=[color], s=120, marker=marker,
                    edgecolors='black', linewidth=1.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Setosa (Actual)'),
        Patch(facecolor='red', alpha=0.7, label='Versicolour (Actual)'),
        Patch(facecolor='lime', alpha=0.7, label='Predicted Setosa'),
        Patch(facecolor='orange', alpha=0.7, label='Predicted Versicolour'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='Correct'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
                   markersize=10, label='Wrong')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.xlabel(f'Feature {feature1_idx + 1} (Petal Length)', fontsize=11)
    plt.ylabel(f'Feature {feature2_idx + 1} (Petal Width)', fontsize=11)
    plt.title('Logistic Regression Decision Boundary', fontsize=12)
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    # 6. 单独预测示例
    print("\n" + "=" * 50)
    print("5. 单独预测示例")
    sample = X_test[0].reshape(1, -1)
    pred = model.predict(sample)
    proba = model.predict_proba(sample)
    species = 'Setosa' if pred[0] == 0 else 'Versicolour'
    print(f"样本特征 (标准化后): {X_test[0]}")
    print(f"预测类别: {pred[0]} ({species})")
    print(f"预测概率: Setosa={1 - proba[0]:.4f}, Versicolour={proba[0]:.4f}")

    # 7. 用户输入预测
    print("\n" + "=" * 50)
    print("6. 用户输入预测")
    print("请输入4个特征值 (萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度):")
    print("示例: 5.1 3.5 1.4 0.2")

    try:
        user_input = input("请输入4个数值，用空格分隔: ").strip().split()
        if len(user_input) == 4:
            features = np.array([float(x) for x in user_input]).reshape(1, -1)
            features_scaled = (features - mean_vals) / scale_vals
            pred = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)
            species = 'Setosa' if pred[0] == 0 else 'Versicolour'
            print(f"\n预测结果: {species}")
            print(f"概率: Setosa={1 - proba[0]:.4f}, Versicolour={proba[0]:.4f}")

            # 在图上标记用户输入的点
            print("\n提示: 用户输入的点用绿色星号标记")
            plt.figure(figsize=(8, 6))
            plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm', levels=[-0.1, 0.5, 1.1])

            # 绘制训练数据
            colors = ['blue' if label == 0 else 'red' for label in y]
            plt.scatter(X_orig[:, feature1_idx], X_orig[:, feature2_idx],
                        c=colors, s=50, edgecolors='black', alpha=0.6)

            # 标记用户输入的点
            plt.scatter(features[0, feature1_idx], features[0, feature2_idx],
                        c='green', s=200, marker='*', edgecolors='black',
                        linewidth=2, label=f'Your Input: {species}')

            plt.xlabel(f'Feature {feature1_idx + 1} (Petal Length)')
            plt.ylabel(f'Feature {feature2_idx + 1} (Petal Width)')
            plt.title(f'Your Input Predicted as: {species}')
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.show()
        else:
            print("请输入4个数值！")
    except ValueError as e:
        print(f"输入格式错误: {e}")
        print("请确保输入4个有效的数值！")


if __name__ == "__main__":
    main()