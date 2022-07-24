import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

def model_1():
    data = loadmat('ex3data1.mat')
    print('-----打印出数据展示-----')
    print(data)

    def loaddata(path):
        data=loadmat(path)
        X = data['X']
        y = data['y']
        return X, y

    X, y = loaddata('ex3data1.mat')
    print(X.shape,y.shape,np.unique(y))

    def plot_random_100_image(X):
        sample_index = np.random.choice(np.arange(X.shape[0]), 100)
        sample_images = X[sample_index, :]
        fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
        for row in range(10):
            for column in range(10):
                ax_array[row, column].matshow(sample_images[10 * row + column].reshape((20, 20)),cmap='gray_r')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def Cost(theta, X, y, l):
        theta_reg = theta[1:]
        first = (-y * np.log(sigmoid(X @ theta))) + (y - 1) * np.log(1 - sigmoid(X @ theta))
        reg = (theta_reg @ theta_reg) * l / (2 * len(X))
        return np.mean(first) + reg

    def gradient(theta, X, y, lam):
        theta_reg = theta[1:]
        grand = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
        # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
        reg = np.concatenate([np.array([0]), (lam / len(X)) * theta_reg])
        return grand + reg

    def one_vs_all(X, y, lam, num_labels):  # num_labels = 10,标签数
        all_theta = np.zeros((num_labels, X.shape[1]))
        # 大θ矩阵包含θ1-theta10维度为(10, 401)
        for i in range(1, num_labels + 1):  # 获取每一个数字的y向量
            theta = np.zeros(X.shape[1])  # 求θ_i
            y_i = np.array([1 if label == i else 0 for label in y])  # 逐个训练theta向量
            final_result = minimize(fun=Cost, x0=theta, args=(X, y_i, lam), method='TNC',jac=gradient, options={'disp': True})
            all_theta[i - 1, :] = final_result.x  # 将求好的θi赋值给大θ矩阵
        return all_theta
    X1, y1 = loaddata('ex3data1.mat')
    X = np.insert(X1, 0, 1, axis=1)  # 在第0列，插入数字1，axis控制维度为列(5000, 401)
    y = y1.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）
    all_theta = one_vs_all(X, y, 1, 10)
    print(all_theta) # 每一行是一个分类器的一组参数

    def predict(X, all_theta):
        h = sigmoid(X @ all_theta.T) # 注意的这里的all_theta需要转置
        h_argmax = np.argmax(h, axis=1) # 找到每行的最大索引（0-9）
        h_argmax = h_argmax + 1 # +1后变为每行的分类器 （1-10）
        return h_argmax

    h_argmax=predict(X,all_theta)
    print(h_argmax.shape)
    y_predict = predict(X, all_theta)
    accuracy = np.mean(y_predict == y)
    print('accuracy = {0}%'.format(accuracy * 100))
    print(classification_report(y, y_predict))
    return None

def model_2():
    data = loadmat('ex3data1.mat')
    x = data['X']
    y = data['y']
    x = np.insert(x, 0, values=1, axis=1)
    print('x.shape=',x.shape)
    y = y.flatten()
    print('y.shape=',y.shape)
    theta = loadmat('ex3weights.mat')
    theta1 = theta['Theta1']
    print('theta1.shape=',theta1.shape)
    theta2 = theta['Theta2']
    print('theta2.shape=',theta2.shape)

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # 输入层
    a1=x
    print('a1.shape=',a1.shape)
    # 隐藏层
    z2 = x @ theta1.T
    print('z2.shape=',z2.shape)
    a2 = sigmoid(z2)
    print('a2.shape=',a2.shape)
    # 输出层
    a2 = np.insert(a2, 0, values=1, axis=1)
    print('a2.shape=',a2.shape)
    z3 = a2 @ theta2.T
    print('z3.shape=',z3.shape)
    a3=sigmoid(z3)
    print('a3.shape=',a3.shape)
    y_pre = np.argmax(a3, axis=1)
    y_pre = y_pre + 1
    acc = np.mean(y_pre == y)
    print('accuracy = {}%'.format(acc * 100))
    report=classification_report(y_pre,y)
    print(report)

def model_3():
    data = loadmat('ex4data1.mat')
    weight = loadmat('ex4weights.mat')
    X = data['X']
    y = data['y']
    print('X.shape=',X.shape)
    print('y.shape=',y.shape)
    encoder=OneHotEncoder(sparse=False)
    y_onehot=encoder.fit_transform(y)
    print('y_onehot=\n',y_onehot)
    print('y_onehot.shape=',y_onehot.shape)
    Theta1, Theta2 = weight['Theta1'], weight['Theta2']

    #sigmoid函数
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    #前向传播函数
    def foward_propagate(X,theta1,theta2):
       m=X.shape[0] #m=5000
       a1=np.insert(X,0,values=np.ones(m),axis=1)
       z2=a1*theta1.T
       a2=np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
       z3=a2*theta2.T
       h=sigmoid(z3)
       return a1,z2,a2,z3,h

    #代价函数
    def cost(param,input_size,hide_size,label_num,X,y,learning_rate):
        m=X.shape[0]
        X=np.matrix(X)
        y=np.matrix(y)
        # 将参数数组解开为每个层的参数矩阵，reshape重新定义维度
        theta1=np.matrix(np.reshape(param[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
        theta2=np.matrix(np.reshape(param[hide_size*(input_size+1):],(label_num,(hide_size+1))))
        a1,z2,a2,z3,h=foward_propagate(X,theta1,theta2)
        J=0
        for i in range(m):
            first=np.multiply(-y[i,:],np.log(h[i,:]))
            second=np.multiply((1-y[i,:]),np.log((1-h[i,:])))
            J=J+np.sum(first-second)
        J=J/m
        return J

    def cost_1(Theta1, Theta2, input_size, hide_size, num_labels, X, y, learning_rate):
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)
        a1, z2, a2, z3, h = foward_propagate(X, Theta1, Theta2)
        J = 0
        for i in range(m):
            first = np.multiply(-y[i, :], np.log(h[i, :]))
            second = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
            J = J + np.sum(first - second)
        J = J / m
        return J

    #初始化设置
    input_size=400
    hide_size=25
    label_num=10
    learning_rate=1
    param=(np.random.random(size=hide_size*(input_size+1)+label_num*(hide_size+1))-0.5)*0.25
    # np.random.random生成（-0.5~0.5）*0.25的浮点型随机数组
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    theta1 = np.matrix(np.reshape(param[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(param[hide_size * (input_size + 1):], (label_num, (hide_size + 1))))
    print('theta1.shape=',theta1.shape)
    print('theta2.shape=',theta2.shape)
    a1,z2,a2,z3,h=foward_propagate(X,theta1,theta2)
    print('a1.shape=',a1.shape)
    print('z2.shape=',z2.shape)
    print('a2.shape=',a2.shape)
    print('z3.shape=',z3.shape)
    print('h.shape=',h.shape)
    print('cost=',cost(param,input_size,hide_size,label_num,X,y_onehot,learning_rate))
    print('cost_1=',cost_1(Theta1, Theta2, input_size, hide_size, label_num, X, y_onehot, learning_rate))

    #正则化代价函数
    def reg_cost(param,input_size,hide_size,label_num,X,y,learning_rate):
        m=X.shape[0]
        X=np.matrix(X)
        y=np.matrix(y)
        theta1=np.matrix(np.reshape(param[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(param[hide_size * (input_size + 1):], (label_num, (hide_size + 1))))
        a1, z2, a2, z3, h = foward_propagate(X, theta1, theta2)
        J = 0
        for i in range(m):
            first = np.multiply(-y[i, :], np.log(h[i, :]))
            second = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
            J =J+ np.sum(first - second)
        J = J / m
        J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
        return J

    def reg_cost_1(Theta1, Theta2, input_size, hide_size, num_labels, X, y, learning_rate):
        J = cost_1(Theta1, Theta2, input_size, hide_size, num_labels, X, y, learning_rate)
        J += (float(learning_rate) / (2 * m)) * (
                    np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2)))
        return J

    print('reg_cost=',reg_cost(param, input_size, hide_size, label_num, X, y_onehot, learning_rate))
    print('reg_cost_1=', reg_cost_1(Theta1, Theta2, input_size, hide_size, label_num, X, y_onehot, learning_rate))

    # 计算我们之前创建的Sigmoid函数的梯度的函数。
    def sigmoid_gradient(z):
        return np.multiply(sigmoid(z),(1-sigmoid(z)))

    def back_propagate(param,input_size,hide_size,label_num,X,y,learning_rate):
        m=X.shape[0]
        X=np.matrix(X)
        y=np.matrix(y)
        theta1 = np.matrix(np.reshape(param[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(param[hide_size * (input_size + 1):], (label_num, (hide_size + 1))))
        a1, z2, a2, z3, h = foward_propagate(X, theta1, theta2)
        J = 0
        delta1=np.zeros(theta1.shape)
        delta2=np.zeros(theta2.shape)
        for i in range(m):
            first=np.multiply(-y[i,:],np.log(h[i,:]))
            second=np.multiply((1-y[i,:]),np.log((1-h[i,:])))
            J=J+np.sum(first-second)
        J=J/m
        J+=(float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
        for t in range(m):
            a1_t=a1[t,:]
            z2_t=z2[t,:]
            a2_t=a2[t,:]
            z3_t=z3[t,:]
            h_t=h[t,:]
            y_t=y[t,:]
            d3_t=h_t-y_t
            z2_t = sigmoid_gradient(np.insert(z2_t, 0, values=np.ones(1)))
            d2_t=np.multiply((theta2.T*d3_t.T).T,z2_t)
            delta1=delta1+(d2_t[:,1:]).T*a1_t
            delta2=delta2+d3_t.T*a2_t
        delta1=delta1/m
        delta2=delta2/m
        grad=np.concatenate((np.ravel(delta1),np.ravel(delta2)))
        return J,grad
    J,grad=back_propagate(param,input_size,hide_size,label_num,X,y_onehot,learning_rate)
    print('J=',J)
    print('grad.shape=',grad.shape)

    #梯度函数加正则化
    def reg_back_propagate(param,input_size,hide_size,label_num,X,y,learning_rate):
        m=X.shape[0]
        X=np.matrix(X)
        y=np.matrix(y)
        theta1 = np.matrix(np.reshape(param[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(param[hide_size * (input_size + 1):], (label_num, (hide_size + 1))))
        a1, z2, a2, z3, h = foward_propagate(X, theta1, theta2)
        J = 0
        delta1=np.zeros(theta1.shape)
        delta2=np.zeros(theta2.shape)
        for i in range(m):
            first=np.multiply(-y[i,:],np.log(h[i,:]))
            second=np.multiply((1-y[i,:]),np.log((1-h[i,:])))
            J=J+np.sum(first-second)
        J=J/m
        J+=(float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
        for t in range(m):
            a1_t=a1[t,:]
            z2_t=z2[t,:]
            a2_t=a2[t,:]
            z3_t=z3[t,:]
            h_t=h[t,:]
            y_t=y[t,:]
            d3_t=h_t-y_t
            z2_t = sigmoid_gradient(np.insert(z2_t, 0, values=np.ones(1)))
            d2_t=np.multiply((theta2.T*d3_t.T).T,z2_t)
            delta1=delta1+(d2_t[:,1:]).T*a1_t
            delta2=delta2+d3_t.T*a2_t
        delta1=delta1/m
        delta2=delta2/m
        #添加正则项
        delta1[:,1:]=delta1[:,1:]+(theta1[:,1:]*learning_rate)/m
        delta2[:,1:]=delta2[:,1:]+(theta2[:,1:]*learning_rate)/m
        grad=np.concatenate((np.ravel(delta1),np.ravel(delta2)))
        return J,grad
    J_reg, grad_reg = reg_back_propagate(param, input_size, hide_size, label_num, X, y_onehot, learning_rate)
    print('J_reg=',J_reg)
    print('grad_reg.shape=',grad_reg.shape)

    #进行预测
    fmin=minimize(fun=reg_back_propagate,x0=param,args=(input_size,hide_size,label_num,X,y_onehot,learning_rate),method='TNC',jac=True,options={'maxiter':250})
    print('fmin=\n',fmin)

    X = np.matrix(X)
    theta1 = np.matrix(np.reshape(fmin.x[:hide_size * (input_size + 1)], (hide_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hide_size * (input_size + 1):], (label_num, (hide_size + 1))))
    a1, z2, a2, z3, h = foward_propagate(X, theta1, theta2)
    y_predict = np.array(np.argmax(h, axis=1) + 1)
    print('y_predict=\n',y_predict)

    accuracy = np.mean(y_predict == y)
    print('accuracy = {}%'.format(accuracy * 100))
    print(classification_report(y, y_predict))

if __name__=='__main__':
    model_1()