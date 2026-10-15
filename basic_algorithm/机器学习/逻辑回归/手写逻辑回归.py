import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import database.read_data as rd

# ---------------------- 1. 数据加载与预处理 ----------------------
def load_iris_data():
    x,y = rd.ReadIris().binary_classification();
    return x, y

# ================= 2. 单独激活函数方法 =================
def sigmoid(z):
    """Sigmoid 激活函数，将线性输出映射到 (0, 1)"""
    z = np.clip(z,-500, 500) # clip 防止溢出
    return 1 / (1 + np.exp(-z))


# ================= 3. 单独代价损失方法 =================
def compute_cost(X, y, w, b):
    """计算二元交叉熵损失 (Binary Cross-Entropy)"""
    h = sigmoid(X.dot(w) + b)
    # 防止 log(0) 导致 NaN
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


# ================= 4. 单独梯度下降方法 =================
def gradient_descent(X, y, w, b, learning_rate=0.1, iterations=1000):
    """
    梯度下降训练
    注意：参数单独计算梯度，单独更新
    """
    cost_history = []
    for i in range(iterations):
        # 前向传播
        z = X.dot(w) + b
        h = sigmoid(z)

        # 误差
        error = h - y  # shape: (100, 1)

        # 单独计算梯度
        # dw 的形状: (n_features, 1)
        dw = (1 / m) * X.T.dot(error)
        # db 是一个标量
        db = (1 / m) * np.sum(error)

        # 单独更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录代价
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost:.6f}")

    return w, b, cost_history

# ================= 1. 读取数据与预处理 =================
# 取前100条数据 (Setosa 和 Versicolor)
X_raw,y = load_iris_data()

# 特征标准化（逻辑回归强烈建议标准化，以加速收敛）
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std

m, n_features = X.shape

# ================= 5. 训练数据 (使用全部4个特征) =================
print("开始训练逻辑回归模型 (4个特征)...")
w_init = np.zeros((n_features, 1))
b_init = 0.0

w_final, b_final, costs = gradient_descent(X, y, w_init, b_init)
print("训练完成！\n")


# ================= 6. 输入4个特征，预测是哪种花 =================
def predict_flower(features_raw, w, b, X_mean, X_std):
    """
    输入原始4个特征，预测花的种类
    features_raw: 1x4 的数组
    """
    # 必须使用训练时的均值和标准差进行标准化
    features_scaled = (features_raw - X_mean) / X_std
    prob = sigmoid(features_scaled.dot(w) + b)[0][0]
    a = sigmoid(features_scaled.dot(w) + b)
    return "Iris-setosa" if prob < 0.5 else "Iris-versicolor"


# 测试预测
test_sample_1 = np.array([[5.1, 3.5, 1.4, 0.2]])  # 典型的 Setosa
test_sample_2 = np.array([[6.0, 2.7, 4.5, 1.5]])  # 典型的 Versicolor

print(f"样本1预测结果: {predict_flower(test_sample_1, w_final, b_final, X_mean, X_std)}")
print(f"样本2预测结果: {predict_flower(test_sample_2, w_final, b_final, X_mean, X_std)}\n")

# ================= 7. 挑出两个特征画模拟图 =================
# 选择 Petal Length (索引2) 和 Petal Width (索引3) 作为绘图特征
feat_x_idx, feat_y_idx = 2, 3

# 为了在2D平面上画出4D模型的决策边界，我们将另外两个特征固定在平均值
# 在标准化空间中，平均值就是 0
X_plot_mean = np.zeros(n_features)
X_plot_mean[0] = 0  # 占位
X_plot_mean[1] = 0

# 生成 2D 网格
x_min, x_max = X[:, feat_x_idx].min() - 1, X[:, feat_x_idx].max() + 1
y_min, y_max = X[:, feat_y_idx].min() - 1, X[:, feat_y_idx].max() + 1
horizontal = np.linspace(x_min, x_max, 5) #从最小值到最大值生成100个数
vertical = np.linspace(y_min, y_max, 5)
xx, yy = np.meshgrid(horizontal,vertical) #平铺网格

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_4d = np.zeros((grid_points.shape[0], n_features))
grid_4d[:, feat_x_idx] = grid_points[:, 0]
grid_4d[:, feat_y_idx] = grid_points[:, 1]

# 计算网格点的预测概率
Z = sigmoid(grid_4d.dot(w_final) + b_final).reshape(xx.shape)

# 绘图
plt.figure(figsize=(10, 7))

# levels
#  levels =[0.5] — 只画"Z = 0.5"的那条线
#  levels=5 均匀分5层，画5条线
contour = plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)

# 绘制原始数据散点
plt.scatter(X[y.ravel() == 0, feat_x_idx], X[y.ravel() == 0, feat_y_idx],
            color='blue', edgecolor='k', label='Iris-setosa (True)', s=60)
plt.scatter(X[y.ravel() == 1, feat_x_idx], X[y.ravel() == 1, feat_y_idx],
            color='red', edgecolor='k', label='Iris-versicolor (True)', s=60)

# 绘制模型预测结果（用不同形状表示，如三角形）
# 这里我们故意把预测为1但真实为0的点（误分类）用特殊形状标出
predictions = (Z > 0.5).astype(int)
# 为了演示，我们在训练集上找出预测错误的点
train_preds = (sigmoid(X.dot(w_final) + b_final) > 0.5).astype(int).ravel()
misclassified_idx = np.where(train_preds != y.ravel())[0]

if len(misclassified_idx) > 0:
    plt.scatter(X[misclassified_idx, feat_x_idx], X[misclassified_idx, feat_y_idx],
                marker='^', color='orange', edgecolor='k', s=100, label='Misclassified')

# 手动创建图例，避免 QuadContourSet 警告
decision_line = Line2D([0], [0], color='green', linewidth=2, label='Decision Boundary')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(decision_line)
labels.append('Decision Boundary')
plt.legend(handles=handles, loc='upper left')

plt.title('Logistic Regression Decision Boundary (4 Features Trained, 2 Features Plotted)', fontsize=14)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# np.meshgrid 作用
# x = np.linspace(1,3,3).astype(int) #从1开始，到3结束，均匀生成3个数 （包括开始和结束）
# y = np.linspace(4,6,3).astype(int) #从4开始，到6结束，均匀生成3个数 （包括开始和结束）
# xx,yy = np.meshgrid(x,y) 平铺二维网格，x按照一行一行的铺，y按照一列一列的铺。
# print(xx)                如果获取平面[1,2]坐标的值，xx[1,2]=3,yy[1,2]=6,平面的[1,2]坐标的值就是(3,6)
# print(yy)
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
# ----------------------------------
# [[4 4 4]
#  [5 5 5]
#  [6 6 6]]