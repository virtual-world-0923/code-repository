import numpy as np
import matplotlib.pyplot as plt

#特征缩放
def featureScaling(x):
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
def gradDesc(alpha = 0.01,theta=[[0],[0],[0]],iter=1000 ):
    cost_histtories= np.zeros(iter)
    for i in range(iter):
        h = logisticRegression(X.dot(theta))
        error = h - y
        deltaTheta = 1.0/m * (X.T.dot(error))
        theta -= alpha * deltaTheta
    return theta
#画决策图和决策边界
def showLine(X,y,theta):
    for i in range(m):
        if (y[i]==0):
            plt.plot(X[i,1],X[i,2],'rx')
        if (y[i]==1):
            plt.plot(X[i,1],X[i,2],'b*')

        min_x = min(X[:, 1])
        max_x = max(X[:, 1])
        y_min_x = (-theta[0] - theta[1] * min_x) / theta[2]  # 这是令sigmoid函数的输入为零，得到分界处的值
        y_max_x = (-theta[0] - theta[1] * max_x) / theta[2]
        plt.plot([min_x, max_x], [y_min_x, y_max_x], 'go--')
    plt.show()

#测试模型精度
def testModel(testX,testY,theta):
    s = 0
    for i in range(m):
        h = logisticRegression(testX[i].dot(theta))
        if (np.where(h>0.5 ,1, 0)==y[i]):
            s+=1
    return s/m
#读取数据
data = np.loadtxt('ex2data1.txt',delimiter=',')
m = data.shape[0]
x = data[:,:-1]
y = data[:,-1].reshape(-1,1)
X = featureScaling(x)
theta = gradDesc()
print(theta)
showLine(X,y,theta)
rate = testModel(X,y,theta)
print(rate)