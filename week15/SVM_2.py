# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib as mpl
"""
改进SMO算法以加快我们的SVM运行速度！
"""

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
类说明：维护所有需要操作的值
Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率   
Returns:
    None
"""
"""
定义一个新的数据结构
"""
class opt_struct:
    def __init__(self, data_mat_in, class_labels, C, toler):
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
    fXk = float(np.multiply(os.alphas, os.label_mat).T * (os.X * os.X[k, :].T) + os.b)
    # 计算误差项
    Ek = fXk - float(os.label_mat[k])
    # 返回误差项
    return Ek


"""
函数说明：内循环启发方式2
选择第二个待优化的alpha_j，选择一个误差最大的alpha_j
即，我们在选择alpha_2的时候做了改进，选择误差最大的
Parameters:
    i - 标号为i的数据的索引值
    oS - 数据结构
    Ei - 标号为i的数据误差 
Returns:
    j - 标号为j的数据的索引值
    maxK - 标号为maxK的数据的索引值
    Ej - 标号为j的数据误差
"""
def select_j(i, os, Ei):
    # 初始化
    max_K = -1 #下标的索引值
    max_delta_E = 0
    Ej = 0
    # 根据Ei更新误差缓存，即先计算alpha_1以及E1值
    os.ecache[i] = [1, Ei] #放入缓存当中，设为有效
    # 对一个矩阵.A转换为Array类型
    # 返回误差不为0的数据的索引值
    valid_ecache_list = np.nonzero(os.ecache[:, 0].A)[0] #找出缓存中不为0
    # 有不为0的误差
    if(len(valid_ecache_list) > 1):
        # 遍历，找到最大的Ek
        for k in valid_ecache_list: #迭代所有有效的缓存，找到误差最大的E
            # 不计算k==i节省时间
            if k == i: #不选择和i相等的值
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
        # 返回max_K，Ej
        return max_K, Ej #这样我们就得到误差最大的索引值和误差最大的值
    # 没有不为0的误差
    else: #第一次循环时是没有有效的缓存值的，所以随机选一个（仅会执行一次）
        # 随机选择alpha_j的索引值
        j = select_j_random(i, os.m)
        # 计算Ej
        Ej = cal_Ek(os, j)
    # 返回j，Ej
    return j, Ej

"""
函数说明：计算Ek,并更新误差缓存
Parameters:
    os - 数据结构
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
函数说明：优化的SMO算法
Parameters:
    i - 标号为i的数据的索引值
    os - 数据结构
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
        j, Ej = select_j(i, os, Ei) #这里不再是随机选取了
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
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - os.X[j, :] * os.X[j, :].T
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
        b1 = os.b - Ei - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[i, :].T - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[i, :].T
        b2 = os.b - Ej - os.label_mat[i] * (os.alphas[i] - alpha_i_old) * os.X[i, :] * os.X[j, :].T - os.label_mat[j] * (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if(0 < os.alphas[i] < os.C):
            os.b = b1
        elif(0 < os.alphas[j] < os.C):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1 #表示有更新
    else:
        return 0 #表示没有更新

"""
函数说明：完整的线性SMO算法
Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    oS.b - SMO算法计算的b
    oS.alphas - SMO算法计算的alphas
"""
def smo_p(data_mat_in, class_labels, C, toler, max_iter):
    # 初始化数据结构
    os = opt_struct(np.mat(data_mat_in), np.mat(class_labels).transpose(), C, toler)
    # 初始化当前迭代次数
    iter = 0
    entrie_set = True #是否在全部数据集上迭代
    alpha_pairs_changed = 0
    # 遍历整个数据集alpha都没有更新或者超过最大迭代次数，则退出循环
    while(iter < max_iter) and ((alpha_pairs_changed > 0) or (entrie_set)):
        alpha_pairs_changed = 0
        if entrie_set: # 遍历整个数据集
            for i in range(os.m):
                # 使用优化的SMO算法
                alpha_pairs_changed += innerL(i, os) #innerL返回的值是0或者1
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            non_bound_i_s = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in non_bound_i_s:
                alpha_pairs_changed += innerL(i, os)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历一次后改为非边界遍历
        if entrie_set:
            entrie_set = False #进行切换，遍历非边界数据集
        # 如果alpha没有更新，计算全样本遍历
        elif(alpha_pairs_changed == 0):
            entrie_set = True
        print("迭代次数:%d" % iter)
    # 返回SMO算法计算的b和alphas
    return os.b, os.alphas

"""
函数说明：分类结果可视化
Returns:
    dataMat - 数据矩阵
    classLabels - 数据标签
    w - 直线法向量
    b - 直线截距 
Returns:
    None
"""
def show_classifer(data_mat, class_labels, w, b):
    data_mat, label_mat = load_dataset('testSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm = mpl.colors.ListedColormap(['g', 'b'])
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
        if (abs(alpha) > 0):
            x, y = data_mat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()
"""
函数说明：计算w
Returns:
    dataArr - 数据矩阵
    classLabels - 数据标签
    alphas - alphas值
Returns:
    w - 直线法向量
"""
def cal_w_s(alphas, data_array, class_labels):
    X = np.mat(data_array)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w


if __name__ == '__main__':
    data_array, class_labels = load_dataset('testSet.txt')
    b, alphas = smo_p(data_array, class_labels, 0.6, 0.001, 40)
    w = cal_w_s(alphas, data_array, class_labels)
    print('b=', b)
    print('w=', w)
    print('alphas=', alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0.0:
            print(data_array[i], class_labels[i])
    show_classifer(data_array, class_labels, w, b)
    