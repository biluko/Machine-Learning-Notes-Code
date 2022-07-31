import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from numpy import *
import pandas as pd
"""
PCA对二维数据进行降维
"""

def load_dataset():
    path='./data/ex7data1.mat'
    two_dimension_data=loadmat(path)
    X=two_dimension_data['X']
    return X

def plot_scatter(X):
    plt.figure(figsize=(12,8))
    cm=mpl.colors.ListedColormap(['blue'])
    plt.scatter(X[:,0],X[:,1],cmap=cm)
    plt.show()

"""
对X去均值,并可视化图像
"""
def demean(X):
    X_demean=(X-mean(X,axis=0))
    plt.figure(figsize=(12,8))
    plt.scatter(X_demean[:,0],X_demean[:,1])
    plt.show()
    return X_demean

"""
计算协方差矩阵
"""
def sigma_matrix(X_demean):
    sigma=(X_demean.T @ X_demean)/X_demean.shape[0]
    return sigma

"""
计算特征值、特征向量
"""
def usv(sigma):
    u,s,v=linalg.svd(sigma)
    return u,s,v

def project_data(X_demean, u, k):
    u_reduced = u[:,:k]
    z=dot(X_demean, u_reduced)
    return z

def recover_data(z, u, k):
    u_reduced = u[:,:k]
    X_recover=dot(z, u_reduced.T)
    return X_recover

if __name__=='__main__':
    X=load_dataset()
    print(X)
    print('=='*50)
    print(X.shape)
    print('=='*50)
    plot_scatter(X)
    X_demean=demean(X)
    sigma=sigma_matrix(X_demean)
    print(sigma)
    print('=='*50)
    u, s, v=usv(sigma)
    print(u)
    print('=='*50)
    print(s)
    print('=='*50)
    print(v)
    print('=='*50)
    z = project_data(X_demean, u, 1)
    print(z)
    print('=='*50)
    X_recover = recover_data(z, u, 1)
    print(X_recover)
    print('=='*50)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X_demean[:,0],X_demean[:,1])
    ax.scatter(list(X_recover[:, 0]), list(X_recover[:, 1]),c='r')
    ax.plot([X_demean[:,0],list(X_recover[:, 0])],[X_demean[:,1],list(X_recover[:, 1])])
    plt.show()