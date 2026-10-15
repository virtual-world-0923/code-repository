import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#特征缩放
def featureScaling(x,m):
    x_mean = np.mean(x,0)
    x_std = np.std(x,0)
    x = x - x_mean
    x = x / x_std
    X = np.hstack((np.ones((m,1)),x))
    return X

#逻辑函数
def logisticRegression(z):
    model = (1.0)/(1+np.exp(-z))
    return model
#梯度下降
def gradDesc(m,alpha = 0.01,theta=[[0],[0],[0],[0]],iter=1000 ):
    cost_histtories= np.zeros(iter)
    for i in range(iter):
        h = logisticRegression(X.dot(theta))
        error = h - y
        deltaTheta = 1.0/m * (X.T.dot(error))
        theta -= alpha * deltaTheta
    return theta
#画决策图和决策边界
def showLine(X,y,theta):
    # 假设 theta 已经是一维数组 (4,)，且包含截距 theta[0]
    # 假设 X 是原始数据 (2017, 4)，第0列是1，第1,2,3列是特征

    # 1. 确保 theta 维度正确
    if theta.ndim > 1:
        theta = theta.flatten()

    # 2. 创建绘图网格
    # 取 Feature 1 和 Feature 2 的最小最大值作为范围
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
    x2_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 10)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # 3. 【关键步骤】计算平面的高度 (Feature 3)
    # 公式推导：theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x3 = 0
    # 移项得：x3 = -(theta[0] + theta[1]*x1 + theta[2]*x2) / theta[3]
    X3_grid = -(theta[0] + theta[1] * X1_grid + theta[2] * X2_grid) / theta[3]

    # 4. 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画散点 (使用之前的 mask 方法)
    mask = (y.ravel() == 0)
    ax.scatter(X[mask, 1], X[mask, 2], X[mask, 3], c='r', marker='x', label='Class 0')
    ax.scatter(X[~mask, 1], X[~mask, 2], X[~mask, 3], c='b', marker='*', label='Class 1')

    # 画平面
    # alpha=0.5 设置透明度，color='g' 设置绿色
    ax.plot_surface(X1_grid, X2_grid, X3_grid, alpha=0.5, color='g')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    plt.show()

#测试模型精度
def testModel(testX,testY,theta,m):
    s = 0
    for i in range(m):
        h = logisticRegression(testX[i].dot(theta))
        if (np.where(h>0.5 ,1, 0)==testY[i]):
            s+=1
    return s/m
#读取数据
data = np.loadtxt('train.txt',delimiter=',')
testData = np.loadtxt('test.txt',delimiter=',')
m ,n= data.shape[0],testData.shape[0]
x ,testX = data[:,:-1],testData[:,:-1]
y ,testY= data[:,-1].reshape(-1,1),testData[:,-1].reshape(-1,1)
X = featureScaling(x,m)

#print(testX)

testX = featureScaling(testX,n)
theta = gradDesc(m)
#print(theta)
showLine(X,y,theta)
rate = testModel(testX,testY,theta,n)
print(rate)