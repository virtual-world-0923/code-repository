import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#引用多项式特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.c_[np.arange(0, 1, 0.002)]
y = np.c_[stats.norm.rvs(size=500, scale=0.1)]
Y = np.c_[y + X**2]

# print(X.shape)
# print(Y.shape)
poly = PolynomialFeatures(2)
# poly.fit(X)
# polyX = poly.transform(X)
polyX = poly.fit_transform(X) #这一步等于上面两步，计算+转换.推荐这样使用
linear = LinearRegression()
linear.fit(polyX,Y)
pred_y = linear.predict(polyX)
plt.scatter(X,Y)
plt.plot(X,pred_y,color='red')
plt.show()