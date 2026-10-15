import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# ========== 1. 加载数据 ==========
iris = load_iris()
X = iris.data  # (150, 4)
y = iris.target  # (150,)

print("=" * 50)
print("鸢尾花数据集信息")
print("=" * 50)
print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")
print(f"类别数: {len(np.unique(y))}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# ========== 2. 数据预处理 ==========
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化（与手写代码保持一致）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")

# ========== 3. 创建神经网络（sklearn版本）==========
"""
网络结构：
- 输入层: 4个特征
- 隐藏层1: 3个神经元
- 隐藏层2: 2个神经元  
- 隐藏层3: 5个神经元
- 输出层: 3个类别（多分类）
"""
mlp = MLPClassifier(
    hidden_layer_sizes=(3, 2, 5),  # 三个隐藏层：3, 2, 5个神经元
    activation='logistic',  # sigmoid激活函数
    solver='sgd',  # 随机梯度下降
    alpha=0.0,  # L2正则化系数
    batch_size='auto',  # 自动批量大小
    learning_rate='constant',  # 固定学习率
    learning_rate_init=0.5,  # 初始学习率
    max_iter=5000,  # 最大迭代次数
    shuffle=True,  # 是否打乱数据
    random_state=42,  # 随机种子
    tol=1e-6,  # 收敛阈值
    verbose=False,  # 不打印训练过程
    warm_start=False,  # 不重用之前的解
    momentum=0.0,  # 不使用动量
    nesterovs_momentum=False,  # 不使用Nesterov动量
    early_stopping=False,  # 不使用早停
    validation_fraction=0.1,  # 验证集比例
    beta_1=0.9,  # Adam参数（不影响SGD）
    beta_2=0.999,  # Adam参数（不影响SGD）
    epsilon=1e-08,  # Adam参数（不影响SGD）
    n_iter_no_change=10  # 早停迭代次数
)

print("\n" + "=" * 50)
print("神经网络结构（sklearn MLPClassifier）")
print("=" * 50)
print(f"隐藏层结构: {mlp.hidden_layer_sizes}")
print(f"激活函数: {mlp.activation}")
print(f"优化器: {mlp.solver}")
print(f"学习率: {mlp.learning_rate_init}")
print(f"最大迭代次数: {mlp.max_iter}")

# ========== 4. 训练模型 ==========
print("\n开始训练...")
mlp.fit(X_train_scaled, y_train)

print(f"\n实际迭代次数: {mlp.n_iter_}")
print(f"最终损失值: {mlp.loss_:.6f}")

# ========== 5. 输出权重维度 ==========
print("\n" + "=" * 50)
print("每一层的权重维度")
print("=" * 50)

# coefs_ 包含所有层的权重矩阵
# intercepts_ 包含所有层的偏置
for i, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    layer_name = f"层{i + 1}"
    if i == 0:
        layer_name += " (输入→隐藏层1)"
    elif i == len(mlp.coefs_) - 1:
        layer_name += f" (隐藏层{len(mlp.coefs_) - 1}→输出层)"
    else:
        layer_name += f" (隐藏层{i}→隐藏层{i + 1})"

    print(f"\n{layer_name}:")
    print(f"  权重矩阵形状: {coef.shape} (行: 上一层神经元数, 列: 当前层神经元数)")
    print(f"  偏置向量形状: {intercept.shape}")

    # 显示具体的维度信息
    if i == 0:
        print(f"  说明: {coef.shape[0]}个输入特征 → {coef.shape[1]}个神经元")
    elif i == len(mlp.coefs_) - 1:
        print(f"  说明: {coef.shape[0]}个神经元 → {coef.shape[1]}个输出类别")
    else:
        print(f"  说明: {coef.shape[0]}个神经元 → {coef.shape[1]}个神经元")

print("\n" + "=" * 50)

# ========== 6. 预测 ==========
print("\n预测测试集...")
y_pred = mlp.predict(X_test_scaled)
y_pred_proba = mlp.predict_proba(X_test_scaled)  # 获取概率

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 详细的分类报告
print("\n" + "=" * 50)
print("分类报告")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ========== 7. 展示预测概率 ==========
print("\n" + "=" * 50)
print("测试集预测概率示例 (前5个样本)")
print("=" * 50)
for i in range(min(5, len(X_test))):
    print(f"\n样本 {i + 1}:")
    print(f"  真实类别: {iris.target_names[y_test[i]]} ({y_test[i]})")
    print(f"  预测类别: {iris.target_names[y_pred[i]]} ({y_pred[i]})")
    print(f"  各类别概率:")
    for j, class_name in enumerate(iris.target_names):
        print(f"    {class_name}: {y_pred_proba[i, j]:.4f}")
    print(f"  预测置信度: {np.max(y_pred_proba[i, :]):.4f}")

# ========== 8. 与手写代码对比 ==========
print("\n" + "=" * 50)
print("与手写神经网络对比")
print("=" * 50)
print("手写代码 vs sklearn MLPClassifier")
print(f"网络结构:     相同 (4→3→2→5→3)")
print(f"激活函数:     相同 (sigmoid)")
print(f"优化算法:     相同 (SGD)")
print(f"学习率:        相同 (0.5)")
print(f"迭代次数:      相同 (5000)")
print(f"权重初始化:    不同 (Xavier vs 默认)")
print(f"数据标准化:    相同 (StandardScaler)")
print(f"损失函数:      相同 (交叉熵)")

# ========== 9. 额外的sklearn特色功能 ==========
print("\n" + "=" * 50)
print("sklearn MLPClassifier的额外功能")
print("=" * 50)
print(f"1. 是否收敛: {mlp.n_iter_ < mlp.max_iter}")
print(f"2. 训练损失历史: 已存储 (可访问 mlp.loss_curve_)")
print(f"3. 验证分数: 已计算 (可访问 mlp.validation_scores_)")
print(f"4. 支持热启动: {mlp.warm_start}")
print(f"5. 支持早停: {mlp.early_stopping}")

# 可选：打印损失曲线（如果存储了）
if hasattr(mlp, 'loss_curve_'):
    print(f"\n损失曲线长度: {len(mlp.loss_curve_)}")
    print(f"初始损失: {mlp.loss_curve_[0]:.6f}")
    print(f"最终损失: {mlp.loss_curve_[-1]:.6f}")

# ========== 10. 模型参数保存与加载示例 ==========
print("\n" + "=" * 50)
print("模型参数示例 (前3个权重值)")
print("=" * 50)
print("第1层权重 (前3个值):")
print(mlp.coefs_[0].flatten()[:3])
print("第1层偏置 (前3个值):")
print(mlp.intercepts_[0][:3])

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)