# -*- coding: utf-8 -*-
import numpy as np
import random
import operator
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
            #print("L == H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j] #这里的计算就要采用核函数了
        if eta >= 0:
            #print("eta >= 0")
            return 0
        # 步骤4：更新alpha_j
        os.alphas[j] -= os.label_mat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        # 更新Ej至误差缓存
        update_Ek(os, j)
        if(abs(os.alphas[j] - alpha_j_old) < 0.00001):
            #print("alpha_j变化太小")
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
                #print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
            iter += 1
        # 遍历非边界值
        else:
            # 遍历不在边界0和C的alpha
            non_nound_i_s = np.nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in non_nound_i_s:
                alpha_pairs_changed += innerL(i, os)
                #print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_pairs_changed))
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
函数说明：将32*32的二进制图像转换为1*1024向量
Parameters:
    filename - 文件名
Returns:
    returnVect - 返回二进制图像的1*1024向量
"""
def imgage_vector(file_name):
    # 创建1*1024零向量
    return_vector = np.zeros((1, 1024))
    # 打开文件
    fr = open(file_name)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        line_string = fr.readline()
        # 每一行的前32个数据依次存储到returnVect中
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_string[j])
    # 返回转换后的1*1024向量
    return return_vector


"""
函数说明：加载图片
Parameters:
    dirName - 文件夹名字
Returns:
    trainingMat - 数据矩阵
    data_labels - 数据标签
"""
def load_images(dir_name):
    from os import listdir
    # 测试集的Labels
    data_labels = []
    # 返回trainingDigits目录下的文件名
    training_file_list = listdir(dir_name)
    # 返回文件夹下文件的个数
    m = len(training_file_list)
    # 初始化训练的Mat矩阵（全零阵），测试集
    training_mat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        file_name_string = training_file_list[i]
        file_string = file_name_string.split('.')[0]
        # 获得分类的数字
        class_number = int(file_string.split('_')[0])
        if class_number == 9:
            data_labels.append(-1)
        else:
            data_labels.append(1)
        training_mat[i, :] = imgage_vector('%s/%s' % (dir_name, file_name_string))
    return training_mat, data_labels

"""
函数说明：测试函数
Parameters:
    kTup - 包含核函数信息的元组
Returns:
    None
"""
def test_digits(kTup=('rbf', 10)):
    # 加载训练集
    data_array, label_array = load_images('trainingDigits')
    # 根据训练集计算b, alphas
    b, alphas = smo_p(data_array, label_array, 200, 0.001, 10, kTup)
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array).transpose()
    # 获得支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = data_mat[svInd]
    labelSV = label_mat[svInd]
    print("支持向量个数:%d" % np.shape(sVs)[0])
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        # 计算各个点的核
        kernel_eval = kernel_trans(sVs, data_mat[i, :], kTup)
        # 根据支持向量的点计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 返回数组中各元素的正负号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    # 打印错误率
    print('训练集错误率:%.2f%%' % (float(error_count) / m))
    # 加载测试集
    data_array, label_array = load_images('testDigits')
    error_count = 0
    data_mat = np.mat(data_array)
    label_mat = np.mat(label_array).transpose()
    m, n = np.shape(data_mat)
    for i in range(m):
        # 计算各个点的核
        kernel_eval = kernel_trans(sVs, data_mat[i, :], kTup)
        # 根据支持向量的点计算超平面，返回预测结果
        predict = kernel_eval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 返回数组中各元素的正负号，用1和-1表示，并统计错误个数
        if np.sign(predict) != np.sign(label_array[i]):
            error_count += 1
    # 打印错误率
    print('测试集错误率:%.2f%%' % (float(error_count) / m))

"""
KNN算法
"""
def classify(test_dataset,train_dataset,labels,k):
    """
    :param test_dataset:用于分类的数据（测试集）
    :param train_dataset:用于训练的数据（训练集）
    :param labels:分类标签
    :param k:选择距离最小的k个点
    :return:
    """
    # numpy函数shape[0]返回dataset的行数
    dataset_size=train_dataset.shape[0]
    # tile具有重复功能，dataset_size是重复四遍，后面的1保证重复完了是四行，而不是一行里有四个是一样的
    diff_mat=np.tile(test_dataset,(dataset_size,1))-train_dataset
    # 二次特征相减后平方
    sqr_diff_mat=diff_mat**2
    # sum()所有元素相加，sum(0）列相加，sum(1)行相加
    sq_distance=sqr_diff_mat.sum(axis=1)
    # 开方，计算出距离
    distance=sq_distance**0.5
    # 返回distance中元素从小到大排列后的索引值
    sorted_distance=distance.argsort()
    # 定一个记录类别次数的字典
    class_count={}
    for i in range(k):
        # 取出前k个元素的类别
        label=labels[sorted_distance[i]]
        # dict.get(key,default=None)，字典的get()方法，返回指定键的值，如果值不在字典中返回默认值
        # 计算类别次数
        class_count[label]=class_count.get(label,0)+1
        # python3中用items()替换python2中的iteritems()
        # key=operator.iteritems(1)根据字典的值进行排序
        # key=operator.iteritems(0)根据字典的键进行排序
        # reverse降序排列字典
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    # 返回次数最多的类别
    return sorted_class_count[0][0]

def hand_writing_class_test():
    import os
    data_labels = []
    # 导入训练集，‘trainingDigits’是一个文件夹
    training_file_list = os.listdir('trainingDigits')
    # 计算训练样本个数
    m = len(training_file_list)
    # 初始化数据集，将所有训练数据用一个m行，1024列的矩阵表示
    training_mat = np.zeros((m,1024))
    for i in range(m):
        # 获得所有文件名，文件名格式‘x_y.txt’,x表示这个手写数字实际表示的数字（label）
        file_name_string = training_file_list[i]
        # 去除 .txt
        file_str = file_name_string.split('.')[0]
        # classnumber为每个样本的分类，用‘_’分割，取得label
        class_num_str = int(file_str.split('_')[0])
        # 将所有标签都存进hwLables[]
        data_labels.append(class_num_str)
        # 将文件转化为向量后存入trainingMat[],这里展现了灵活的文件操作
        training_mat[i,:] = imgage_vector('./trainingDigits/%s' % file_name_string)
    test_file_list = os.listdir('testDigits')  #迭代测试集
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_string = test_file_list[i]
        # 去除 .txt
        file_str = file_name_string.split('.')[0]
        class_num_string = int(file_str.split('_')[0])
        # 这部分针对测试集的预处理和前面基本相同
        vector_under_test = imgage_vector('./testDigits/%s' % file_name_string)
        # 使用算法预测样本所属类别，调用了前面写的classify0()函数
        classify_result = classify(vector_under_test, training_mat, data_labels, 3)
        # print ("分类结果:%d,真实类别:%d" % (classify_result, class_num_string))
        # 算法结果与样本的实际分类做对比
        if (classify_result != class_num_string): error_count += 1.0
    print('KNN算法的预测情况:')
    print ("\n分类错误的个数为: %d" % error_count)
    print ("\n分类错误率为: %f" % (error_count/float(m_test)))

if __name__ == '__main__':
    test_digits()
    hand_writing_class_test()
