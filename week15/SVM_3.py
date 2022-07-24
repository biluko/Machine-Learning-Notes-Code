# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
from matplotlib.patches import Circle
"""
函数说明：读取数据
Parameters:
    file_name - 文件名 
Returns:
    data_mat - 数据矩阵
    label_mat - 数据标签
"""
def load_dataset(file_name):
    # 数据矩阵
    data_mat = []
    # 标签向量
    label_mat = []
    # 打开文件
    fr = open(file_name)
    # 逐行读取
    for line in fr.readlines():
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        # 将每一行内容根据'\t'符进行切片
        line_array = line.strip().split('\t')
        # 添加数据(100个元素排成一行)
        data_mat.append([float(line_array[0]), float(line_array[1])])
        # 添加标签(100个元素排成一行)
        label_mat.append(float(line_array[2]))
    return data_mat, label_mat

"""
函数说明：通过核函数将数据转换更高维空间
Parameters:
    X - 数据矩阵
    A - 单个数据的向量
    kTup - 包含核函数信息的元组 
Returns:
    K - 计算的核K
"""
def kernel_trans(xi, xj, kTup):
    """
    :param kTup: 两维，第一列是字符串，为lin,rbf，如果是rbf，第二列多一个sigmma
    :return:
    """
    # 读取X的行列数
    m, n = np.shape(xi)
    # K初始化为m行1列的零向量
    K = np.mat(np.zeros((m, 1)))
    # 线性核函数只进行内积
    if kTup[0] == 'lin':
        K = xi * xj.T
    # 高斯核函数，根据高斯核函数公式计算
    elif kTup[0] == 'rbf':
        for j in range(m):
            delta_row = xi[j, :] - xj
            K[j] = delta_row * delta_row.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('核函数无法识别')
    return K

"""
类说明：维护所有需要操作的值
Parameters:
    data_mat_in - 数据矩阵
    class_cabels - 数据标签
    C - 松弛变量
    toler - 容错率   
Returns:
    None
"""
"""
定义一个新的数据结构
"""
class opt_struct:
    def __init__(self, data_mat_in, class_labels, C, toler,kTup):
        # 数据矩阵
        self.X = data_mat_in #传进来的数据
        # 数据标签
        self.label_mat = class_labels
        # 松弛变量
        self.C = C
        # 容错率
        self.tol = toler
        # 矩阵的行数
        self.m = np.shape(data_mat_in)[0]
        # 根据矩阵行数初始化alphas矩阵，一个m行1列的全零列向量
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 初始化b参数为0
        self.b = 0
        # 根据矩阵行数初始化误差缓存矩阵，第一列为是否有效标志位，其中0无效，1有效;第二列为实际的误差Ei的值
        """
        我们之前的定义为:Ei=gxi-yi
        yi是标签的实际值。
        gx=alpha_i*y_i*x_i.x,就相当于是w.x+b
        因为误差值经常用到，所以希望每次计算后放到一个缓存当中，将ecache一分为二，第一列是标志位，取值为0或者1，为1时表示已经算出来
        """
        self.ecache = np.mat(np.zeros((self.m, 2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernel_trans(self.X,self.X[i,:],kTup)

"""
函数说明：计算误差
Parameters:
    os - 数据结构
    k - 标号为k的数据
Returns:
    Ek - 标号为k的数据误差
"""
def cal_Ek(os, k):
    # multiply(a,b)就是个乘法，如果a,b是两个数组，那么对应元素相乘
    # .T为转置
    fXk = float(np.multiply(os.alphas, os.label_mat).T * os.K[:, k]+os.b)
    # 计算误差项
    Ek = fXk - float(os.label_mat[k])
    # 返回误差项
    return Ek

"""
函数说明：随机选择alpha_j
Parameters:
    i - alpha_i的索引值
    m - alpha参数个数
Returns:
    j - alpha_j的索引值
"""
def select_j_random(i, m):
    j = i
    while(j == i):
        # uniform()方法将随机生成一个实数，它在[x, y)范围内
        j = int(random.uniform(0, m))
    return j

"""
函数说明：内循环启发方式2
Parameters:
    i - 标号为i的数据的索引值
    os - 数据结构
    Ei - 标号为i的数据误差
Returns:
    j - 标号为j的数据的索引值
    max_K - 标号为maxK的数据的索引值
    Ej - 标号为j的数据误差
"""
def select_j(i, os, Ei):
    # 初始化
    max_K = -1
    max_delta_E = 0
    Ej = 0
    # 根据Ei更新误差缓存
    os.ecache[i] = [1, Ei]
    # 对一个矩阵.A转换为Array类型
    # 返回误差不为0的数据的索引值
    valid_ecache_list = np.nonzero(os.ecache[:, 0].A)[0]
    # 有不为0的误差
    if(len(valid_ecache_list) > 1):
        # 遍历，找到最大的Ek
        for k in valid_ecache_list:
            # 不计算k==i节省时间
            if k == i:
                continue
            # 计算Ek
            Ek = cal_Ek(os, k)
            # 计算|Ei - Ek|
            delta_E = abs(Ei - Ek)
            # 找到maxDeltaE
            if(delta_E > max_delta_E):
                max_K = k
                max_delta_E = delta_E
                Ej = Ek
        # 返回maxK，Ej
        return max_K, Ej
    # 没有不为0的误差
    else:
        # 随机选择alpha_j的索引值
        j = select_j_random(i, os.m)
        # 计算Ej
        Ej = cal_Ek(os, j)
    # 返回j，Ej
    return j, Ej

"""
函数说明：计算Ek,并更新误差缓存
Parameters:
    oS - 数据结构
    k - 标号为k的数据的索引值 
Returns:
    None
"""
def update_Ek(os, k):
    # 计算Ek
    Ek = cal_Ek(os, k)
    # 更新误差缓存
    os.ecache[k] = [1, Ek]

"""
函数说明：修剪alpha_j
Parameters:
    aj - alpha_j值
    H - alpha上限
    L - alpha下限 
Returns:
    aj - alpha_j值
"""
def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
函数说明：优化的SMO算法
Parameters:
    i - 标号为i的数据的索引值
    oS - 数据结构 
Returns:
    1 - 有任意一对alpha值发生变化
    0 - 没有任意一对alpha值发生变化或变化太小
"""
def innerL(i, os):
    # 步骤1：计算误差Ei
    Ei = cal_Ek(os, i)
    # 优化alpha,设定一定的容错率
    if((os.label_mat[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or ((os.label_mat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = select_j(i, os, Ei)
        # 保存更新前的alpha值，使用深层拷贝
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        # 步骤2：计算上界H和下界L
        if(os.label_mat[i] != os.label_mat[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L == H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j] #这里的计算就要采用核函数了
        if eta >= 0:
            print("eta >= 0")
            return 0
        # 步骤4：更新alpha_j
        os.alphas[j] -= os.label_mat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        # 更新Ej至误差缓存
        update_Ek(os, j)
        if(abs(os.alphas[j] - alpha_j_old) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        os.alphas[i] += os.label_mat[i] * os.label_mat[j] * (alpha_j_old - os.alphas[j])
        # 更新Ei至误差缓存
        update_Ek(os, i)
        # 步骤7：更新b_1和b_2:
        b1 = os.b - Ei - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.K[i, i] - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.K[j, i]
        b2 = os.b - Ej - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.K[i, j] - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if(0 < os.alphas[i] < os.C):
            os.b = b1
        elif(0 < os.alphas[j] < os.C):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

"""
函数说明：完整的线性SMO算法
Parameters:
    data_mat_in - 数据矩阵
    class_labels - 数据标签
    C - 松弛变量
    toler - 容错率
    max_iter - 最大迭代次数
    kTup - 包含核函数信息的元组 
Returns:
    os.b - SMO算法计算的b
    os.alphas - SMO算法计算的alphas
"""
def smo_p(data_mat_in, class_labels, C, toler, max_iter, kTup = ('lin', 0)):
    # 初始化数据结构
    os = opt_struct(np.mat(data_mat_in), np.mat(class_labels).transpose(), C, toler, kTup)
    # 初始化当前迭代次数
    iter = 0
    entrie_set = True
    alpha_pairs_changed = 0
    # 遍历整个数据集alpha都没有更新或者超过最大迭代次数，则退出循环
    while(iter < max_iter) and ((alpha_pairs_changed > 0) or (entrie_set)):
        alpha_pairs_changed = 0
        # 遍历整个数据集
        if entrie_set:
            for i in range(os.m):
                # 使用优化的SMO算法
                alpha_pairs_changed += innerL(i, os)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            non_nound_i_s = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in non_nound_i_s:
                alpha_pairs_changed += innerL(i, os)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历一次后改为非边界遍历
        if entrie_set:
            entrie_set = False
        # 如果alpha没有更新，计算全样本遍历
        elif(alpha_pairs_changed == 0):
            entrie_set = True
        print("迭代次数:%d" % iter)
    # 返回SMO算法计算的b和alphas
    return os.b, os.alphas

"""
函数说明：测试函数
Parameters:
    k1 - 使用高斯核函数的时候表示到达率  
Returns:
    None
"""
def test_rbf(k1 = 1.3):
    # 加载训练集
    data_array, label_array = load_dataset('testSetRBF.txt')
    # 根据训练集计算b, alphas
    b, alphas = smo_p(data_array, label_array, 200, 0.0001, 100, ('rbf', k1))
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array).transpose()
    # 获得支持向量
    svInd = np.nonzero(alphas.A > 0)[0] # 从行维度来描述索引值
    sVs = data_mat[svInd]
    labelSV = label_mat[svInd]
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        # 计算各个点的核
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ('rbf', k1))
        # 根据支持向量的点计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 返回数组中各元素的正负号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(label_array[i]):
            error_count += 1
    # 打印错误率
    print('训练集错误率:%.2f%%' % ((float(error_count) / m) * 100))
    # 加载测试集
    data_array, label_array = load_dataset('testSetRBF2.txt')
    error_count = 0
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array).transpose()
    m, n = np.shape(data_mat)
    for i in range(m):
        # 计算各个点的核
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ('rbf', k1))
        # 根据支持向量的点计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 返回数组中各元素的正负号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(label_array[i]):
            error_count += 1
    # 打印错误率
    print('测试集错误率:%.2f%%' % ((float(error_count) / m) * 100))

def cal_w_s(alphas, data_mat, class_labels):
    X = np.mat(data_mat)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w

"""
函数说明：数据可视化
Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签  
Returns:
    None
"""
def show_dataset(data_mat, label_mat):
    for path in ['testSetRBF.txt','testSetRBF2.txt']:
        data_mat, label_mat = load_dataset(path)
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['b', 'g'])
    ax.scatter(np.array(data_mat)[:, 0], np.array(data_mat)[:, 1], c=np.array(label_mat).squeeze(), cmap=cm_dark, s=30)
    b, alphas = smo_p(data_mat, label_mat, 200, 0.0001, 10000, ('rbf', 2))
    w = cal_w_s(alphas, data_mat, label_mat)
    # 画支持向量
    alphas_non_zeros_index = np.where(alphas > 0)
    for i in alphas_non_zeros_index[0]:
        circle = Circle((data_mat[i][0], data_mat[i][1]), 0.035, facecolor='none', edgecolor='red', linewidth=1.5,alpha=1)
        ax.add_patch(circle)
    plt.show()

if __name__ == '__main__':
    for path in ['testSetRBF.txt','testSetRBF2.txt']:
        data_mat, label_mat = load_dataset(path)
        print('=='*60)
        show_dataset(data_mat,label_mat)
        print('=='*60)
    test_rbf()