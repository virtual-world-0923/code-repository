import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import database.read_data as rd

# ============================================================
# 1、读取数据（前100条，二分类）
# ============================================================
X, y = rd.ReadIris().binary_classification()

# ============================================================
# 2、训练模型（使用4个特征）
# ============================================================
model = LogisticRegression(
    solver='saga',
    max_iter=5000
)
model.fit(X, y)

print("训练完成")
print("模型权重:", np.round(model.coef_, 4))
print("偏置:", np.round(model.intercept_, 4))

# ============================================================
# 3、预测（输入4个特征）
# ============================================================
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(sample)
prob = model.predict_proba(sample)

print("预测类别:", pred[0])
print("预测概率:", np.round(prob[0], 4))

# ============================================================
# 4、取两个特征画二维图
# ============================================================
feature1 = X[:, 0]  # 萼片长度
feature2 = X[:, 2]  # 萼片宽度

class_0 = y == 0
class_1 = y == 1

plt.figure(figsize=(7, 5))

# 训练数据
plt.scatter(feature1[class_0.ravel()], feature2[class_0.ravel()],
            color='red', label='Setosa', alpha=0.7)
plt.scatter(feature1[class_1.ravel()], feature2[class_1.ravel()],
            color='blue', label='Versicolor', alpha=0.7)

# 分割线：w1*x1 + w2*x2 + b = 0
w = model.coef_[0][:2]
b = model.intercept_[0]

x1_line = np.linspace(feature1.min() - 0.5, feature1.max() + 0.5, 100)
x2_line = -(b + w[0] * x1_line) / w[1]

plt.plot(x1_line, x2_line, 'k--', linewidth=2, label='Decision Boundary')

# 预测样本（不同形状 & 颜色）
test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.5, 3.0, 5.5, 1.8],
    [5.8, 2.6, 4.0, 1.2]
])

for ts in test_samples:
    p = model.predict(ts.reshape(1, -1))[0]
    marker = 'X' if p == 0 else 'P'
    color = 'lime' if p == 0 else 'gold'

    plt.scatter(ts[0], ts[1],
                color=color, marker=marker, s=250,
                edgecolors='black', linewidths=1.5,
                label=f'Pred={p}')

# 去重 legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Logistic Regression (sklearn)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()