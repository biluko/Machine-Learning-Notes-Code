import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
np.random.seed(0)#用于生成指定随机数。
"""
seed()被设置了之后，np,random.random()可以按顺序产生一组固定的数组，
如果使用相同的seed()值，则每次生成的随机数都相同，
如果不设置这个值，那么每次生成的随机数不同。
但是，只在调用的时候seed()一下并不能使生成的随机数相同，需要每次调用都seed()一下，表示种子相同，从而生成的随机数相同。
"""
x=np.sort(5*np.random.rand(40,1),axis=0)#axis = 0 按行计算,得到列的性质。 axis = 1 按列计算,得到行的性质。
T=np.linspace(0,5,500)[:,np.newaxis]#一维数组转化为shape（1，500）的数组
y=np.sin(x).ravel()#numpy.ravel()展平函数,变为一维
y[::5] += 1 * (0.5 - np.random.rand(8))
n_neighbors = 5
for i, weights in enumerate(["uniform", "distance"]):
    # enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    knn = KNeighborsRegressor(n_neighbors, weights=weights)
    y_predict = knn.fit(x, y).predict(T)
    plt.subplot(2, 1, i+1)
    plt.scatter(x, y, color="darkorange", label="data")
    plt.plot(T, y_predict, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))
plt.tight_layout()
plt.show()

