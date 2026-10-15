import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import database.read_data as rd


class NeuralNetwork:
    def __init__(self, layer_dims):
        """
        初始化神经网络参数
        layer_dims: 列表，包含每层的神经元数量 [input, hidden1, hidden2, hidden3, output]
        """
        self.layer_dims = layer_dims
        self.params = {}
        self.grads = {}
        self.cache = {}

        # 初始化权重和偏置 (He Initialization for ReLU)
        np.random.seed(42)
        for l in range(1, len(layer_dims)):
            # 权重矩阵形状: (当前层神经元数, 上一层神经元数)
            self.params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0 / layer_dims[l - 1])
            # 偏置向量形状: (当前层神经元数, 1)
            self.params[f'b{l}'] = np.zeros((layer_dims[l], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def softmax(self, Z):
        # 防止溢出
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """
        前向传播过程
        """
        self.cache['A0'] = X
        A = X
        L = len(self.layer_dims) - 1  # 总层数（不含输入层）

        for l in range(1, L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']

            # 线性部分: Z = W.A + b
            Z = np.dot(W, A) + b

            # 激活部分
            if l == L:
                # 输出层使用 Softmax
                A = self.softmax(Z)
            else:
                # 隐藏层使用 ReLU
                A = self.relu(Z)

            # 缓存中间结果用于反向传播
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        return A

    def compute_cost(self, Y, AL):
        """
        单独的代价函数：交叉熵损失
        Y: 真实标签 (One-Hot), 形状 (3, m)
        AL: 预测概率, 形状 (3, m)
        """
        m = Y.shape[1]
        # 添加微小值防止 log(0)
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return cost

    def backward_propagation(self, X, Y, AL):
        """
        反向传播梯度计算过程
        详细展示每一步梯度的推导
        """
        m = X.shape[1]
        L = len(self.layer_dims) - 1

        # 1. 输出层梯度计算 (Softmax + Cross Entropy)
        # 对于 Softmax 输出和 Cross Entropy 损失，dZ_L = AL - Y
        dZ = AL - Y
        self.grads[f'dW{L}'] = (1 / m) * np.dot(dZ, self.cache[f'A{L - 1}'].T)
        self.grads[f'db{L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # 上一层的激活值梯度 dA_{L-1} = W_L.T . dZ_L
        dA_prev = np.dot(self.params[f'W{L}'].T, dZ)

        # 2. 隐藏层梯度反向传播 (从 L-1 到 1)
        for l in reversed(range(1, L)):
            # 应用 ReLU 导数: dZ_l = dA_l * g'(Z_l)
            dZ = dA_prev * self.relu_derivative(self.cache[f'Z{l}'])

            # 计算当前层的权重和偏置梯度
            self.grads[f'dW{l}'] = (1 / m) * np.dot(dZ, self.cache[f'A{l - 1}'].T)
            self.grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # 计算传递给下一层（更靠近输入层）的激活值梯度
            if l > 1:
                dA_prev = np.dot(self.params[f'W{l}'].T, dZ)

    def update_parameters(self, learning_rate):
        """
        单独的梯度下降函数：更新权重和偏置
        """
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            self.params[f'W{l}'] -= learning_rate * self.grads[f'dW{l}']
            self.params[f'b{l}'] -= learning_rate * self.grads[f'db{l}']

    def train(self, X, Y, epochs=1000, learning_rate=0.1):
        """
        训练循环
        """
        for i in range(epochs):
            # 前向传播
            AL = self.forward_propagation(X)

            # 计算代价
            cost = self.compute_cost(Y, AL)

            # 反向传播
            self.backward_propagation(X, Y, AL)

            # 更新参数
            self.update_parameters(learning_rate)

            if i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        """
        预测时最终输出三种类别的概率
        """
        probs = self.forward_propagation(X)
        return probs

    def print_weights_dimensions(self):
        """
        训练完成后输出每一层的权重维度
        """
        print("\n--- Weight Dimensions ---")
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            print(f"Layer {l} Weight (W{l}) shape: {self.params[f'W{l}'].shape}")
            print(f"Layer {l} Bias (b{l}) shape: {self.params[f'b{l}'].shape}")


# --- 主程序执行 ---
# X ,Y_labels = rd.ReadIris().get_data()
# 1. 加载数据
iris = load_iris()
X = iris.data
Y_labels = iris.target

# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T  # 转置为 (features, samples)

# One-Hot 编码标签
encoder = OneHotEncoder(sparse_output=False)
Y_onehot = encoder.fit_transform(Y_labels.reshape(-1, 1)).T  # 转置为 (classes, samples)

# 划分训练集和测试集 (为了演示简单，这里直接使用全部数据训练，实际应用中应划分)
# 这里我们打乱数据并取前100个作为训练示例，或者全量训练
X_train = X_scaled
Y_train = Y_onehot

# 3. 定义网络结构
# 输入层4, 隐藏1(3), 隐藏2(2), 隐藏3(5), 输出层3
layer_dims = [4, 3, 2, 5, 3]

# 4. 实例化模型
nn = NeuralNetwork(layer_dims)

# 5. 训练模型
print("Starting Training...")
nn.train(X_train, Y_train, epochs=1000, learning_rate=0.5)

# 6. 输出权重维度
nn.print_weights_dimensions()

# 7. 预测并输出概率
print("\n--- Prediction Probabilities for first 5 samples ---")
probs = nn.predict_proba(X_train[:, :5])
for i in range(5):
    true_class = np.argmax(Y_train[:, i])
    pred_class = np.argmax(probs[:, i])
    print(f"Sample {i + 1}: True Class={true_class}, Pred Class={pred_class}")
    print(f"  Probabilities: Setosa={probs[0, i]:.4f}, Versicolor={probs[1, i]:.4f}, Virginica={probs[2, i]:.4f}")