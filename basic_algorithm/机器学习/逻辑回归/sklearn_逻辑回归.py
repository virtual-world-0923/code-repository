import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import database.read_data as rd

# ================= 1. 数据加载与预处理 =================
X_raw ,y = rd.Sample_Custom().get_binary_classification()

# sklearn 逻辑回归强烈建议标准化，这里使用库自带的 StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ================= 2. 训练模型 =================
# max_iter 设置大一点防止警告，solver='liblinear' 适合小规模数据
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_scaled, y.ravel())

print("模型训练完成！")
print(f"模型权重 w: {model.coef_}")
print(f"模型偏置 b: {model.intercept_}")

# ================= 3. 预测功能 =================
def predict_flower(features_raw):
    # 必须用训练好的 scaler 进行转换
    features_scaled = scaler.transform(features_raw)
    pred = model.predict(features_scaled)
    return pred

# 测试两个样本
test_sample_1 = [-1.54,3.87]
test_sample_2 = [3.51,1.82]
test_sample = np.vstack([test_sample_1,test_sample_2])
test_pred = predict_flower(test_sample)
print(f"样本1预测: {test_pred[0]}")
print(f"样本2预测: {test_pred[1]}")

# ================= 4. 极简画图 =================

# plt.figure(figsize=(9, 6))

# 1. 画出训练数据：两种不同类别用不同颜色
plt.scatter(X_scaled[y.ravel()==0, 0], X_scaled[y.ravel()==0, 1],
            color='blue', edgecolor='k', label='True: 0', s=60)
plt.scatter(X_scaled[y.ravel()==1, 0], X_scaled[y.ravel()==1, 1],
            color='red', edgecolor='k', label='True: 1', s=60)

# 2. 画出预测数据：用不同的形状（五角星）和颜色
plt.scatter(test_sample[test_pred==0, 0], test_sample[test_pred==0, 1],
            marker='*', color='green', s=150, label='Pred: 0')
plt.scatter(test_sample[test_pred==1, 0], test_sample[test_pred==1, 1],
            marker='*', color='orange', s=150, label='Pred: 1')

# 3. 【核心】画出分割线 (Decision Boundary)
# 提取当前选中的两个特征的权重和偏置
w1 = model.coef_[0][0]
w2 = model.coef_[0][1]
b = model.intercept_[0]

# 线性方程: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
# 生成横轴的极值范围
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
x_vals = np.array([x_min, x_max])
y_vals = -(w1 * x_vals + b) / w2

# 绘制分割线
plt.plot(x_vals, y_vals, 'g--', linewidth=2, label='Decision Boundary')

plt.title('Sklearn Logistic Regression Visualization', fontsize=14)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()