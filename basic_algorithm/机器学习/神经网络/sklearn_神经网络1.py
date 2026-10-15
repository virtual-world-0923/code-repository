from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 4个输入特征
y = iris.target  # 3分类标签 0/1/2

# 2. 数据预处理：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 搭建多层感知机，匹配指定网络结构
# hidden_layer_sizes=(隐1神经元数, 隐2神经元数, 隐3神经元数) = (3, 2, 5)
mlp = MLPClassifier(
    hidden_layer_sizes=(3, 2, 5),  # 3层隐藏层：3 → 2 → 5
    activation="logistic",         # 等价手写代码sigmoid激活
    solver="sgd",                  # 梯度下降优化，和手写GD逻辑对齐
    learning_rate_init=0.12,       # 学习率和手写版本保持一致
    max_iter=10000,                # 最大迭代轮数
    random_state=42,
    tol=1e-6,                      # 损失收敛阈值
    # verbose=True                   # 打印训练损失变化
)

# 5. 模型训练
mlp.fit(X_train, y_train)

# 6. 输出每一层权重与偏置维度（对应需求9）
print("\n========== 各层权重、偏置维度 ==========")
# mlp.coefs_：存储所有层权重矩阵 [W1, W2, W3, W4]
# mlp.intercepts_：存储所有层偏置 [b1, b2, b3, b4]
layer_names = [
    "W1(输入4→隐1(3))",
    "W2(隐1(3)→隐2(2))",
    "W3(隐2(2)→隐3(5))",
    "W4(隐3(5)→输出3分类)"
]
bias_names = [
    "b1(隐1偏置)",
    "b2(隐2偏置)",
    "b3(隐3偏置)",
    "b4(输出层偏置)"
]

for i in range(len(mlp.coefs_)):
    print(f"{layer_names[i]} 维度: {mlp.coefs_[i].shape}")
    print(f"{bias_names[i]} 维度: {mlp.intercepts_[i].shape}")

# 7. 预测输出三分类概率（需求10）
print("\n========== 测试集前5样本预测概率(3类别) ==========")
test_probs = mlp.predict_proba(X_test[:5])  # 直接输出softmax概率
for idx, prob in enumerate(test_probs):
    print(f"真实标签{y_test[idx]}")
    print(f"样本{idx+1} 类别0:{prob[0]:.4f} 类别1:{prob[1]:.4f} 类别2:{prob[2]:.4f}")

# 8. 模型评估准确率
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集分类准确率: {acc:.4f}")

# 9. 查看最终训练损失
# print(f"\n训练完成最终损失值: {mlp.loss_:.4f}")