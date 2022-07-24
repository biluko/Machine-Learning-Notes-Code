import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
def regression():
    path = ('ex1data1.txt')
    data = pd.read_csv(path, header=None, names=['population', 'profit'])
    print('------展示前十行代码------')
    print(data.head(10))
    """
    看下数据长什么样子
    """
    data.plot(kind='scatter', x='population', y='profit', figsize=(10, 6))
    plt.ylabel('profit')
    plt.xlabel('population')
    """
    现在让我们使用梯度下降来实现线性回归，以最小化成本函数。
    """
    def Cost(X, y, theta):
        inner = np.power(((X * theta.T) - y), 2)
        return np.sum(inner) / (2 * len(X))
    """
    让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度
    """
    data.insert(0, 'Ones', 1)
    alpha = 0.01
    iters = 10000
    """
    现在我们来做一些变量初始化
    """
    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
    y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列
    """
    代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
    """
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))
    theta_n = (X.T * X).I * X.T * y
    print('-----利用正规方程求系数-----')
    print(theta_n)
    print('-----初始化theta-----')
    print(theta)
    print('-----检查维数-----')
    print(X.shape, theta.shape, y.shape)
    print('-----计算代价函数的值为-----')
    print(Cost(X, y, theta))
    print('------批量梯度下降------')
    def gradientDescent(X, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])
        cost = np.zeros(iters)
        for i in range(iters):
            error = (X * theta.T) - y
            for j in range(parameters):
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            theta = temp
            cost[i] = Cost(X, y, theta)
        return theta, cost
    """
    现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
    """
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    print('-----随机梯度下降法求得的参数-----')
    print(final_theta)
    """
    最后，我们可以使用我们拟合的参数计算训练模型的代价函数（误差）。
    """
    print('-----拟合的参数计算训练模型的代价函数-----')
    print(Cost(X, y, final_theta))
    """
    绘制线性模型以及数据，直观地看出它的拟合。
    """
    x = np.linspace(data.population.min(), data.population.max(), 100)
    f = final_theta[0, 0] + (final_theta[0, 1] * x)
    a = float(final_theta[0, 0])
    b = float(final_theta[0, 1])
    print('-----输出回归方程-----')
    print('回归方程为:f=%.8f+%.8f*x' % (a, b))
    x_predict1 = np.matrix([[1, 3.5]])
    x_predict2 = np.matrix([[1, 7]])
    predict1 = np.dot(x_predict1, final_theta.T)
    predict2 = np.dot(x_predict2, final_theta.T)
    print('-----输出预测值1-----')
    print(predict1)
    print('-----输出预测值2-----')
    print(predict2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.population, data.profit, label='Data')
    ax.legend(loc=2)
    ax.set_xlabel('population')
    ax.set_ylabel('profit')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(iters), cost, 'r', label='alpha=0.01')
    alpha = 0.001
    theta = np.matrix(np.array([0, 0]))
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters), cost, 'b', label='alpha=0.001')
    alpha = 0.0001
    theta = np.matrix(np.array([0, 0]))
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters), cost, 'black', label='alpha=0.0001')
    alpha = 0.00001
    theta = np.matrix(np.array([0, 0]))
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters), cost, 'yellow', label='alpha=0.00001')
    ax.legend(loc=2)
    ax.set_xlabel('iters')
    ax.set_ylabel('cost')
    ax.set_title('gradient function with different alpha')
    plt.show()
    return None
def regression2():
    path = 'ex1data2.txt'
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print('-----展示前十行数据-----')
    print(data.head(10))
    """
    对于此任务，我们添加了另一个预处理步骤 - 特征归一化。
    """
    new_data = (data - data.mean()) / data.std()
    print('-----特征归一化后的数据-----')
    print(new_data.head(10))
    """
    重复第1部分的预处理步骤，并对新数据集运行线性回归程序。
    """
    new_data.insert(0, 'Ones', 1)
    cols = new_data.shape[1]
    X = new_data.iloc[:, 0:cols - 1]
    y= new_data.iloc[:, cols - 1:cols]
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    print(X.shape,y.shape)
    theta = np.matrix(np.array([0, 0, 0]))
    theta_n = (X.T * X).I * X.T * y
    print('-----利用正规方程求系数-----')
    print(theta_n)
    alpha=0.001
    iters=10000
    def Cost(X, y, theta):
        inner = np.power(((X * theta.T) - y), 2)
        return np.sum(inner) / (2 * len(X))
    def gradientDescent(X, y, theta, alpha, iters):
        temp = np.matrix(np.zeros(theta.shape))
        parameters = int(theta.ravel().shape[1])
        cost = np.zeros(iters)
        for i in range(iters):
            error = (X * theta.T) - y
            for j in range(parameters):
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
            theta = temp
            cost[i] = Cost(X, y, theta)
        return theta, cost
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    print('-----代价函数值-----')
    print(Cost(X, y, final_theta))
    print('-----输出系数值-----')
    print(final_theta)
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2 = np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    f = final_theta[0, 0] + final_theta[0, 1] * x1 + final_theta[0, 2] * x2
    a=float(final_theta[0,0])
    b1=float(final_theta[0,1])
    b2=float(final_theta[0,2])
    print('输出的回归方程为:f=%.8f*x1%.8f*x2'%(b1,b2))
    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, f, rstride=1, cstride=1,cmap=cm.autumn, label='prediction')
    ax.scatter(X[:100, 1], X[:100, 2], y[:100, 0], c='black')
    ax.set_zlabel('PRICE')
    ax.set_ylabel('BATHROOM')
    ax.set_xlabel('SIZE')
    """
    快速查看这一个的训练进程。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(iters), cost, 'r',label='alpha=0.001')
    ax.set_xlabel('iters')
    ax.set_ylabel('cost')
    alpha = 0.0001
    theta = np.matrix(np.array([0, 0,0]))
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters), cost, 'yellow', label='alpha=0.0001')
    alpha = 0.01
    theta = np.matrix(np.array([0, 0, 0]))
    final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
    ax.plot(np.arange(iters), cost, 'black', label='alpha=0.01')
    ax.legend(loc=2)
    ax.set_title('gradient function with different alpha')
    plt.show()
    return None
def logisticregression():
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    print(data.head(10))
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    random_numbers = np.arange(-10, 10, step=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(random_numbers, sigmoid(random_numbers), 'black')
    def cost(theta, X, y):
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(y)
        first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
        second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
        return np.sum(first - second) / (len(X))
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0,0]))
    print(theta)
    print(X.shape, y.shape,theta.shape)
    print('-----输出的代价函数的值为-----')
    print(cost(theta, X, y))
    def gradient(theta, X, y):
        theta = np.matrix(theta)
        X = np.matrix(X)
        y = np.matrix(y)
        parameters = int(theta.ravel().shape[1])
        grad = np.zeros(parameters)
        error = sigmoid(X * theta.T) - y
        for i in range(parameters):
            term = np.multiply(error, X[:, i])
            grad[i] = np.sum(term) / len(X)
        return grad
    print(gradient(theta, X, y))
    res = opt.minimize(fun=cost, x0=theta, jac=gradient, args=(X, y),method='Newton-CG')
    print(res)
    def predict(theta, X):
        probability = sigmoid(X * theta.T)
        return [1 if x >= 0.5 else 0 for x in probability]
    print(res.x)
    final_theta=res.x
    print(final_theta)
    final_theta=np.matrix(final_theta)
    y_predict = predict(final_theta, X)
    from sklearn.metrics import classification_report
    print(classification_report(y, y_predict))
    coef = -(res.x / res.x[2])
    print(coef)
    x = np.arange(130, step=0.1)
    y = coef[0] + coef[1] * x
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='blue', marker='o', label='Admitted Yes')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='red', marker='x', label='Admitted No')
    ax.plot(x,y)
    ax.set_xlim(20, 110)
    ax.set_ylim(20, 110)
    ax.legend(loc=2)
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    ax.set_title('Decision Boundary')
    plt.show()
    return None
def logisticregression2():
    path = 'ex2data2.txt'
    data= pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accept'])
    print(data.head())
    def feature_mapping(x, y, power, as_ndarray=False):
        data = {"f'{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
                }
        if as_ndarray:
            return np.array(pd.DataFrame(data))
        else:
            return pd.DataFrame(data)
    x1 = np.array(data.Test1)
    x2 = np.array(data.Test2)
    new_data = feature_mapping(x1, x2, power=6)
    print(new_data.shape)
    print(new_data.head())
    theta = np.zeros(new_data.shape[1])
    X = feature_mapping(x1, x2, power=6, as_ndarray=True)
    print(X.shape)  # (118, 28)
    y = np.array(data.iloc[:, -1])
    print(y.shape)  # (118,)
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def regularized_cost(theta, X, y, l=1):
        thetaReg = theta[1:]
        first = (-y * np.log(sigmoid(X @ theta))) - (1 - y) * np.log(1 - sigmoid(X @ theta))
        reg = (thetaReg @ thetaReg) * l / (2 * len(X))
        return np.mean(first) + reg
    print(regularized_cost(theta, X, y, l=1))

    def regularized_gradient(theta, X, y, l=1):
        thetaReg = theta[1:]
        first = (X.T @ (sigmoid(X @ theta) - y)) / len(X)
        reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
        return first + reg
    print(regularized_gradient(theta,X,y))
    print('init cost = {}'.format(regularized_cost(theta, X, y)))
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
    print(res)
    def predict(theta, X):
        probability = sigmoid(X @ theta)
        return [1 if x >= 0.5 else 0 for x in probability]  # return a list
    from sklearn.metrics import classification_report
    final_theta = res.x
    y_predict = predict(final_theta, X)
    predict(final_theta, X)
    print(classification_report(y, y_predict))
    positive = data[data['Accept'].isin([1])]
    negative = data[data['Accept'].isin([0])]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.scatter(positive['Test1'], positive['Test2'], marker='o', c='b', label='Accept')
    ax.scatter(negative['Test1'], negative['Test2'], marker='x', c='r', label='Reject')
    ax.legend(loc=2)
    x_label = 'test1'
    x_label = 'test2'
    x = np.linspace(-1, 1.5, 50)
    # 从-1到1.5等间距取出50个数
    xx, yy = np.meshgrid(x, x)
    # 将x里的数组合成50*50=250个坐标
    z = np.array(feature_mapping(xx.ravel(), yy.ravel(), 6))
    z = z @ final_theta
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, 0, colors='black')
    # 等高线是三维图像在二维空间的投影，0表示z的高度为0
    plt.ylim(-.8, 1.2)
    positive = data[data['Accept'].isin([1])]
    negative = data[data['Accept'].isin([0])]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.scatter(positive['Test1'], positive['Test2'], marker='o', c='b', label='Accept')
    ax.scatter(negative['Test1'], negative['Test2'], marker='x', c='r', label='Reject')
    ax.legend(loc=2)
    x_label = 'test1'
    x_label = 'test2'
    plt.show()
if __name__=='__main__':
    logisticregression2()