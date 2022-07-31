"""
给定一个二维的数据集，使用k-means算法进行聚类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib as mpl

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

"""
绘制散点图
"""
def plot_scatter():
    data,dataset=load_dataset()
    plt.figure(figsize=(12,8))
    plt.scatter(dataset['X1'],dataset['X2'],cmap=['b'])
    plt.show()

"""
获得每个样本所属的类别
"""
def get_near_cluster_centroids(X,centroids):
    """
    :param X: 我们的数据集
    :param centroids: 聚类中心点的初始位置
    :return:
    """
    m = X.shape[0] #数据的行数
    k = centroids.shape[0] #聚类中心的行数，即个数
    idx = np.zeros(m) # 一维向量idx，大小为数据集中的点的个数，用于保存每一个X的数据点最小距离点的是哪个聚类中心
    for i in range(m):
        min_distance = 1000000
        for j in range(k):
            distance = np.sum((X[i, :] - centroids[j, :]) ** 2) # 计算数据点到聚类中心距离代价的公式，X中每个点都要和每个聚类中心计算
            if distance < min_distance:
                min_distance = distance
                idx[i] = j # idx中索引为i，表示第i个X数据集中的数据点距离最近的聚类中心的索引
    return idx # 返回的是X数据集中每个数据点距离最近的聚类中心

def compute_centroids(X, idx, k):
    """
    :param X: 数据集
    :param idx: 每个样本所属的类别
    :param k: 类别总数
    :return:
    """
    m, n = X.shape
    centroids = np.zeros((k, n)) # 初始化为k行n列的二维数组，值均为0，k为聚类中心个数，n为数据列数
    for i in range(k):
        indices = np.where(idx == i) # 输出的是索引位置
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids

def k_means(X, initial_centroids, max_iters):
    """
    :param X:
    :param initial_centroids: 初始聚类中心
    :param max_iters: 迭代次数
    :return:
    """
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids_all = []
    centroids_all.append(initial_centroids)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = get_near_cluster_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
        centroids_all.append(centroids)
    return idx, np.array(centroids_all),centroids

def plot_data(centroids_all,idx):
    plt.figure(figsize=(12,8))
    cm=mpl.colors.ListedColormap(['r','g','b'])
    plt.scatter(dataset['X1'],dataset['X2'], c=idx,cmap=cm)
    plt.plot(centroids_all[:,:,0],centroids_all[:,:,1], 'kx--')
    plt.show()

def plot_classify_data(X,idx):
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
    ax.legend()
    plt.show()


def init_centroids(X, k):
    m, n = X.shape
    init_centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        init_centroids[i, :] = X[idx[i], :]
    return init_centroids

if __name__=='__main__':
    data,dataset = load_dataset()
    plot_scatter()
    X=data.get('X')
    centroids_0=np.array([[3, 3], [6, 2], [8, 5]])
    idx, centroids_all,centroids=k_means(X,centroids_0,10)
    plot_data(centroids_all, idx)
    plot_classify_data(X,idx)
    print(init_centroids(X, 3))
    for i in range(5):
        idx, centroids_all, centroids = k_means(X, init_centroids(X, 3), 10)
        plot_data(centroids_all,idx)