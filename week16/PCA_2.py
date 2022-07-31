from numpy import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
faces_data = loadmat('data/ex7faces.mat')
print(faces_data)
X=faces_data['X']
print(X[0].shape)
print(X.shape)

def plot_100_image(X):
    fig,ax=plt.subplots(nrows=10,ncols=10,figsize=(10,10))
    for c in range(10):
        for r in range(10):
            ax[c,r].imshow(X[10*c+r].reshape(32,32).T,cmap='Greys_r')
            ax[c,r].set_xticks([])
            ax[c,r].set_yticks([])
    plt.show()

plot_100_image(X)

def reduce_mean(X):
    X_reduce_mean=X-X.mean(axis=0)
    return X_reduce_mean
X_reduce_mean=reduce_mean(X)

def sigma_matrix(X_reduce_mean):
    sigma=(X_reduce_mean.T @ X_reduce_mean)/X_reduce_mean.shape[0]
    return sigma
sigma=sigma_matrix(X_reduce_mean)

def usv(sigma):
    u,s,v=linalg.svd(sigma)
    return u,s,v
u,s,v=usv(sigma)
print(u)

def project_data(X_reduce_mean, u, k):
    u_reduced = u[:,:k]
    z=dot(X_reduce_mean, u_reduced)
    return z

def recover_data(z, u, k):
    u_reduced = u[:,:k]
    X_recover=dot(z, u_reduced.T)
    return X_recover

z = project_data(X_reduce_mean, u, 100)

X_recover=recover_data(z,u,100)
plot_100_image(X_recover)

