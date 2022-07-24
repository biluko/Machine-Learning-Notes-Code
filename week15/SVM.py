# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 2022
@author: wzk
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

"""
函数说明：读取数据,处理文本数据
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
        label_mat.append(int(line_array[2]))
    return data_mat, label_mat


"""
函数说明：随机选择alpha_j。alpha的选取，随机选择一个不等于i值的j
Parameters:
    i - 第一个alpha的下标
    m - alpha参数个数 
Returns:
    j - 返回选定的数字
"""
def select_J_rand(i, m):
    """
    :param i: 表示alpha_i
    :param m: 表示样本数
    :return:
    """
    j = i
    while(j == i):
        # 如果i和j相等，那么就从样本中随机选取一个，可以认为j就是选择的alpha_2
        # uniform()方法将随机生成一个实数，它在[x, y)范围内
        j = int(random.uniform(0, m))
    return j

"""
函数说明：修剪alpha
Parameters:
    a_j - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    a_j - alpha值
用于调整大于H或小于L的alpha值。
"""
def clip_alpha(a_j, H, L):
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j

"""
函数说明：简化版SMO算法
Parameters:
    data_mat_in - 数据矩阵
    class_labels - 数据标签
    C - 松弛变量
    toler - 容错率
    max_iter - 最大迭代次数 
Returns:
    None
"""
def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
    """
    :param data_mat_in: 相当于我们的x
    :param class_labels: 相当于我们的y
    :param C: 惩罚因子
    :param toler:
    :param max_iter:
    :return:
    """
    # 转换为numpy的mat矩阵存储(100,2)
    data_matrix = np.mat(data_mat_in) #相当于x
    # 转换为numpy的mat矩阵存储并转置(100,1)
    label_mat = np.mat(class_labels).transpose() #相当于y
    # 初始化b参数，统计data_matrix的维度,m:行；n:列
    b = 0
    # 统计dataMatrix的维度,m:100行；n:2列
    m, n = np.shape(data_matrix)
    # 初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代maxIter次
    while(iter_num < max_iter):
        alpha_pairs_changed = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            # multiply(a,b)就是个乘法，如果a,b是两个数组，那么对应元素相乘
            # .T为转置，转置的目的是因为后面的核函数是一个向量。核函数我们最原始的方式，可以直接使用点乘就可以，即x_i.x
            fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            # 误差项计算公式
            Ei = fxi - float(label_mat[i])
            # 优化alpha，设定一定的容错率
            if((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个alpha_i成对比优化的alpha_j
                j = select_J_rand(i, m)
                # 步骤1，计算误差Ej
                fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                # 误差项计算公式
                Ej = fxj - float(label_mat[j])
                # 保存更新前的alpha值，使用深拷贝(完全拷贝)A深层拷贝为B，A和B是两个独立的个体
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 步骤2：计算上下界H和L
                if(label_mat[i] != label_mat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if(L == H):
                    print("L == H")
                    continue
                # 步骤3：计算eta，转置表示相乘没有错误，相当于求的-eta
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                # 步骤5：修剪alpha_j
                alphas[j] = clip_alpha(alphas[j], H, L)
                if(abs(alphas[j] - alpha_j_old) < 0.00001):
                    print("alpha_j变化太小")
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[i, :].T
                b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if(0 < alphas[i] < C):
                    b = b1
                elif(0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alpha_pairs_changed += 1
                # 打印统计信息
                print("第%d次迭代 样本：%d， alpha优化次数：%d" % (iter_num, i, alpha_pairs_changed))
        # 更新迭代次数
        if(alpha_pairs_changed == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数：%d" % iter_num)
    return b, alphas

"""
函数说明：计算w
Returns:
    data_mat - 数据矩阵
    label_mat - 数据标签
    alphas - alphas值
Returns:
    w - 直线法向量
"""
def get_w(data_mat_in, class_labels, alphas):
    """
    :param data_mat: x
    :param label_mat: y
    :param alphas:
    :return:
    """
    data_mat=np.mat(data_mat_in)
    label_mat=np.mat(class_labels).transpose()
    m,n=np.shape(data_mat)
    # 初始化w都为1
    w=np.zeros((n,1))
    # dot()函数是矩阵乘，而*则表示逐个元素相乘
    # w = sum(alpha_i * yi * xi)
    #循环计算
    for i in range(m):
        w+=np.multiply(alphas[i]*label_mat[i],data_mat[i,:].T)
    return w

"""
函数说明：分类结果可视化
Returns:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线截距
Returns:
    None
"""
def show_classifer(data_mat, w, b):
    data_mat, label_mat = load_dataset('testSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(data_mat)[:, 0], np.array(data_mat)[:, 1], c=np.array(label_mat), cmap=cm, s=20)
    # 绘制直线
    x1 = max(data_mat)[0]
    x2 = min(data_mat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    # enumerate在字典上是枚举、列举的意思
    for i, alpha in enumerate(alphas):
        # 支持向量机的点
        if(abs(alpha) > 0):
            x, y = data_mat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()

if __name__ == '__main__':
    data_mat, label_mat = load_dataset('testSet.txt')
    print('数据标签为:\n',label_mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(data_mat)[:, 0], np.array(data_mat)[:, 1], c=np.array(label_mat), cmap=cm, s=20)

    b, alphas = smo_simple(data_mat, label_mat, 0.6, 0.001, 40)
    w = get_w(data_mat, label_mat, alphas)
    print('b=',b)
    print('w=',w)
    print('alphas=',alphas[alphas>0])
    for i in range(100):
        if alphas[i]>0.0:
            print(data_mat[i],label_mat[i])
    x = np.arange(-2.0, 12, 0.1)
    # w0x0+w1x1=0,x1=-w0x0/w1
    y = (-w[0] * x - b) / w[1]
    ax.plot(x,y.reshape(-1,1))
    ax.axis([-2,12,-8.6,7])
    plt.show()
    show_classifer(data_mat,w,b)