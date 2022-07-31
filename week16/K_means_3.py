import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy import *
from IPython.display import Image
def load_picture():
    path='./data/bird_small.png'
    image=plt.imread(path)
    plt.imshow(image)
    plt.show()

def load_data():
    path='./data/bird_small.mat'
    data=loadmat(path)
    return data

def normalizing(A):
    A=A/255.
    A_new=reshape(A,(-1,3))
    return A_new

def get_near_cluster_centroids(X,centroids):
    m = X.shape[0] #数据的行数
    k = centroids.shape[0] #聚类中心的行数，即个数
    idx = zeros(m) # 一维向量idx，大小为数据集中的点的个数，用于保存每一个X的数据点最小距离点的是哪个聚类中心
    for i in range(m):
        min_distance = 1000000
        for j in range(k):
            distance = sum((X[i, :] - centroids[j, :]) ** 2) # 计算数据点到聚类中心距离代价的公式，X中每个点都要和每个聚类中心计算
            if distance < min_distance:
                min_distance = distance
                idx[i] = j # idx中索引为i，表示第i个X数据集中的数据点距离最近的聚类中心的索引
    return idx # 返回的是X数据集中每个数据点距离最近的聚类中心

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = zeros((k, n)) # 初始化为k行n列的二维数组，值均为0，k为聚类中心个数，n为数据列数
    for i in range(k):
        indices = where(idx == i) # 输出的是索引位置
        centroids[i, :] = (sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids

def k_means(A_1,initial_centroids,max_iters):
    m,n=A_1.shape
    k = initial_centroids.shape[0]
    idx = zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = get_near_cluster_centroids(A_1, centroids)
        centroids = compute_centroids(A_1, idx, k)
    return idx, centroids

def init_centroids(X, k):
    m, n = X.shape
    init_centroids = zeros((k, n))
    idx = random.randint(0, m, k)
    for i in range(k):
        init_centroids[i, :] = X[idx[i], :]
    return init_centroids

def reduce_picture():
    initial_centroids = init_centroids(A_new, 16)
    idx, centroids = k_means(A_new, initial_centroids, 10)
    idx_1 = get_near_cluster_centroids(A_new, centroids)
    A_recovered = centroids[idx_1.astype(int), :]
    A_recovered_1 = reshape(A_recovered, (A.shape[0], A.shape[1], A.shape[2]))
    plt.imshow(A_recovered_1)
    plt.show()

if __name__=='__main__':
    load_picture()
    data=load_data()
    print(data.keys())
    A=data['A']
    print(A.shape)
    A_new=normalizing(A)
    print(A_new)
    print(A_new.shape)
    reduce_picture()
