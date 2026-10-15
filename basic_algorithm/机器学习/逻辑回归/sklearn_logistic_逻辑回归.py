import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:,:-1]
y = data[:,-1].ravel()
lr = LogisticRegression()
lr.fit(X,y)
pred_y = lr.predict(X)
#计算预测标签与真实标签之间匹配的比例或数量.normalize=True 模型预测正确的比例.normalize=False 预测正确的样本总数
print(accuracy_score(y,pred_y))
#打印截距
print(lr.intercept_)
#打印系数
print(lr.coef_)