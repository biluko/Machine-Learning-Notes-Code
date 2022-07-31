"""
给定一个二维的数据集，使用k-means算法进行聚类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
"""
导入数据的函数
"""
def load_dataset():
    path='./data/ex7data2.mat'
    # 字典格式 : <class 'dict'>
    data=loadmat(path)
    # data.keys() : dict_keys(['__header__', '__version__', '__globals__', 'X'])
    dataset = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
    return data,dataset
data,dataset=load_dataset()
print('data=\n',data)
"""
绘制散点图
"""
def plot_scatter():
    data,dataset=load_dataset()
    plt.figure(figsize=(12,8))
    plt.scatter(dataset['X1'],dataset['X2'],cmap=['b'])
    plt.show()

def find_centroids(X,centroids):
    idx=[]
    for i in range(X.shape[0]):
        dist=np.linalg.norm((X[i]-centroids),axis=1)
        idx_i=np.argmin(dist)
        idx.append(idx_i)
    return np.array(idx)

centroids_0=np.array([[3,3],[6,2],[8,5]])
idx_0=find_centroids(data.get('X'),centroids_0)

def compute_centroids(X,idx,k):
    centroids=[]
    for i in range(k):
        centroids_i=np.mean(X[idx==i],axis=0)
        centroids.append(centroids_i)
    return np.array(centroids)

centroids=compute_centroids(data.get('X'),idx_0,3)

def k_means(X,centroids,max_iters):
    k=len(centroids)
    centroids_all=[]
    centroids_all.append(centroids_0)
    centroids_i=centroids
    for i in range(max_iters):
        idx_1=find_centroids(X,centroids_i)
        centroids_i=compute_centroids(X,idx_1,k)
        centroids_all.append(centroids_i)
    return np.array(idx_1),np.array(centroids_all)


def plot_data(X,centroids_all,idx):
    plt.figure(figsize=(12,8))
    plt.scatter(X[:,0],X[:,1],c=idx,cmap='rainbow')
    plt.plot(centroids_all[:,:,0],centroids_all[:,:,1],'kx--')
    plt.show()

idx,centroids_all=k_means(data.get('X'),centroids,max_iters=10)
plot_data(data.get('X'),centroids_all,idx)

def init_centroids(X,k):
    index=np.random.choice(len(X),k)
    return X[index]

print(init_centroids(data.get('X'),k=3))
for i in range(5):
    idx,centroids_all=k_means(data.get('X'),init_centroids(data.get('X'),k=3),max_iters=10)
    plot_data(data.get('X'),centroids_all,idx)