import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ====================== 1. 激活函数与导数 ======================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_deriv(a):
    return a * (1 - a)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) #减去z的最大值，防止溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) #分子分母同时减，结果不变


# ====================== 2. 独立代价函数：交叉熵损失 ======================
def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    log_prob = -np.log(y_pred + 1e-8)
    loss = np.sum(y_true * log_prob) / m
    return loss


# ====================== 3. 独立梯度下降更新函数 ======================
def gradient_descent_update(params, grads, lr):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    params["W3"] -= lr * grads["dW3"]
    params["b3"] -= lr * grads["db3"]
    params["W4"] -= lr * grads["dW4"]
    params["b4"] -= lr * grads["db4"]
    return params


# ====================== 4. 前向传播（修复cache完整存储） ======================
def forward_propagation(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    W3, b3 = params["W3"], params["b3"]
    W4, b4 = params["W4"], params["b4"]

    A0 = X
    Z1 = np.dot(A0, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = softmax(Z4)

    # 完整存入所有中间变量，无遗漏
    cache = {
        "A0": A0,
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3,
        "Z4": Z4, "A4": A4
    }
    return A4, cache


# ====================== 5. 反向传播 修复cache读取方式 ======================
def backward_propagation(y_true, cache, params):
    m = y_true.shape[0]
    W4, W3, W2, W1 = params["W4"], params["W3"], params["W2"], params["W1"]

    # 分步单独取值，避免一次性多变量解包KeyError
    A0 = cache["A0"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    A4 = cache["A4"]

    # Step1：输出层梯度 dZ4
    dZ4 = A4 - y_true
    dW4 = np.dot(A3.T, dZ4) / m
    db4 = np.sum(dZ4, axis=0, keepdims=True) / m

    # Step2：第三隐藏层梯度 dZ3
    dA3 = np.dot(dZ4, W4.T)
    dZ3 = dA3 * sigmoid_deriv(A3)
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # Step3：第二隐藏层梯度 dZ2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_deriv(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Step4：第一隐藏层梯度 dZ1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = np.dot(A0.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {
        "dW1": dW1, "db1": db1,
        "dW2": dW2, "db2": db2,
        "dW3": dW3, "db3": db3,
        "dW4": dW4, "db4": db4
    }
    return grads


# ====================== 6. 初始化网络权重 ======================
def init_network_params():
    #初始化参数对于训练结果影响很大
    np.random.seed(42)
    W1 = np.random.randn(4, 3) * 0.1 #np.sqrt(1.0 / 4.0)   np.sqrt(2.0 / 4.0)
    b1 = np.zeros((1, 3))

    W2 = np.random.randn(3, 2) * 0.1 #np.sqrt(1.0 / 3.0)   np.sqrt(2.0 / 3.0)
    b2 = np.zeros((1, 2))

    W3 = np.random.randn(2, 5) * 0.1 #np.sqrt(1.0 / 2.0)   np.sqrt(2.0 / 2.0)
    b3 = np.zeros((1, 5))

    W4 = np.random.randn(5, 3) * 0.1 #np.sqrt(1.0 / 1.0)   np.sqrt(2.0 / 1.0)
    b4 = np.zeros((1, 3))

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
    return params


# ====================== 7. 训练主函数 ======================
def train_neural_net(X, y_onehot, lr=0.1, epochs=8000, print_loss=500):
    params = init_network_params()
    for epoch in range(epochs):
        y_pred, cache = forward_propagation(X, params)
        loss = cross_entropy_loss(y_pred, y_onehot)
        grads = backward_propagation(y_onehot, cache, params)
        params = gradient_descent_update(params, grads, lr)

        if epoch % print_loss == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
    return params


# ====================== 8. 预测函数（输出概率） ======================
def predict_prob(X, params):
    prob, _ = forward_propagation(X, params)
    return prob


def predict_class(X, params):
    prob = predict_prob(X, params)
    return np.argmax(prob, axis=1)


# ====================== 9. 数据加载与预处理 ======================
iris = load_iris()
X_raw = iris.data
y_raw = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y_raw.reshape(-1, 1))

# m = X.shape[0]
# y_onehot = np.zeros((m, 3))
# for i in range(m):
#     y_onehot[i, y_raw[i]] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# ====================== 10. 执行训练 ======================
trained_params = train_neural_net(X_train, y_train, lr=0.25, epochs=1000, print_loss=200)

# ====================== 11. 输出每一层权重维度 ======================
print("\n========== 各层权重&偏置维度 ==========")
print(f"W1 (输入4→隐1 3神经元): {trained_params['W1'].shape}")
print(f"b1 (隐1偏置): {trained_params['b1'].shape}")
print(f"W2 (隐1 3→隐2 2神经元): {trained_params['W2'].shape}")
print(f"b2 (隐2偏置): {trained_params['b2'].shape}")
print(f"W3 (隐2 2→隐3 5神经元): {trained_params['W3'].shape}")
print(f"b3 (隐3偏置): {trained_params['b3'].shape}")
print(f"W4 (隐3 5→输出3分类): {trained_params['W4'].shape}")
print(f"b4 (输出层偏置): {trained_params['b4'].shape}")

# ====================== 12. 预测输出概率 ======================
print("\n========== 测试集前5样本预测概率 ==========")
test_prob = predict_prob(X_test[:5], trained_params)
for idx, p in enumerate(test_prob):
    print(f"真实标签{y_test[idx]}")
    print(f"样本{idx + 1} 类别0概率:{p[0]:.4f} 类别1概率:{p[1]:.4f} 类别2概率:{p[2]:.4f}")

y_test_label = np.argmax(y_test, axis=1)
test_pred = predict_class(X_test, trained_params)
acc = accuracy_score(y_test_label, test_pred)
print(f"\n测试集准确率: {acc:.4f}")