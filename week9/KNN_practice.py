import numpy as np
import operator
import os
"""
parameters:
X:用于分类的数据，即测试集
dataset:用于训练的数据集，即训练集
labels:分类标签
k:kNN算法参数，选择距离最小的k个点
return:
sorted_class_count[0][0]:分类结果
"""
def classify(X,dataset,labels,k):
    dataset_size=dataset.shape[0] #numpy函数shape[0]返回dataset的行数
    #tile具有重复功能，dataset_size是重复四遍，后面的1保证重复完了是四行，而不是一行里有四个是一样的
    diff_mat=np.tile(X,(dataset_size,1))-dataset
    #二次特征相减后平方
    sqr_diff_mat=diff_mat**2
    #sum()所有元素相加，sum(0）列相加，sum(1)行相加
    sq_distance=sqr_diff_mat.sum(axis=1)
    #开方，计算出距离
    distance=sq_distance**0.5
    #返回distance中元素从小到大排列后的索引值
    sorted_distance=distance.argsort()
    #定一个记录类别次数的字典
    class_count={}
    for i in range(k):
         #取出前k个元素的类别
        label=labels[sorted_distance[i]]
        #dict.get(key,default=None)，字典的get()方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别次数
        class_count[label]=class_count.get(label,0)+1
        #python3中用items()替换python2中的iteritems()
        #key=operator.iteritems(1)根据字典的值进行排序
        #key=operator.iteritems(0)根据字典的键进行排序
        #reverse降序排列字典
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别
    return sorted_class_count[0][0]
"""
函数说明：打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力
:parameter
file_name:文件名
returns:
character_matrix:特征矩阵
class_label_vector:分类label向量
"""
def file_matrix(file_name):
    #打开文件
    file=open(file_name)
    #读取文件所有内容
    array_lines=file.readlines()
    #得到文件的行数
    number_lines=len(array_lines)
    #返回numpy矩阵，解析完成的数据:number_lines行，3列
    return_character_matrix=np.zeros((number_lines,3))
    #返回的分类标签向量
    class_label_vector=[]
    #行的索引值
    index=0
    for line in array_lines:
        #s.strip(rm),当rm为空时，默认删除空白符（包括'\n','\t','\r','')
        line=line.strip()
        #使用s.strip(str='',num=string,count(str))将字符串根据'\t'分隔符进行切片
        list_from_line=line.split('\t')
        #将数据前三列提取出来，存放到character_matrix的numpy矩阵中
        return_character_matrix[index,:]=list_from_line[0:3]
        #根据文本中标记的喜欢程度进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力
        if list_from_line[-1]=='didntLike':
            class_label_vector.append(1)
        elif list_from_line[-1]=='smallDoses':
            class_label_vector.append(2)
        elif list_from_line[-1]=='largeDoses':
            class_label_vector.append(3)
        index+=1
    return return_character_matrix,class_label_vector
"""
函数说明:对数据进行归一化
:parameter
character_matrix:特征矩阵
returns:
norm_dataset:归一化后的特征矩阵
ranges:数据范围
min_vals:数据的最小值
"""
def auto_norm(dataset):
    #获取数据的最小值
    min_vals=dataset.min(0)
    max_vals=dataset.max(0)
    #最大值和最小值的范围
    ranges=max_vals-min_vals
    norm_dataset=np.zeros(np.shape(dataset))
    # 返回特征矩阵的行数
    m = dataset.shape[0]
    norm_dataset=dataset-np.tile(min_vals,(m,1))
    #除以最大和最小值的差，得到归一化数据
    norm_dataset=norm_dataset/np.tile(ranges,(m,1))
    #返回归一化数据结果,数据范围，最小值
    return norm_dataset,ranges,min_vals
"""
函数说明：分类器测试函数
Parameters:
无
returns:
norm_dataset:归一化后的特征矩阵
ranges:数据范围
min_vals:数据最小值
"""
def dating_class_test():
    #打开文件名
    file_name="./2022-06-08data/datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到dating_mat和dating_labels中
    dating_mat,dating_labels=file_matrix(file_name)
    #读取所有数据的10%
    data_rate=0.10
    #数据归一化，返回归一化后的矩阵，数据范围，数据最小值
    norm_mat,ranges,min_vals=auto_norm(dating_mat)
    #获取norm_matrix的行数
    m=norm_mat.shape[0]
    #10%的测试数据的数量
    number_test_data=int(m*data_rate)
    #分类错误计数
    error_number=0.0
    for i in range(number_test_data):
        #前number_test_data个数据作为测试集，后m-number_test_data个数据作为训练集
        classify_result=classify(norm_mat[i,:],norm_mat[number_test_data:m,:],dating_labels[number_test_data:m],3)
        print('分类结果:%d\t真实类别:%d'%(classify_result,dating_labels[i]))
        if classify_result!=dating_labels[i]:
            error_number+=1.0
    print('错误分类:%f%%'%(error_number/float(number_test_data)*100))

def classify_person():
    file_name='./2022-06-08data/datingTestSet.txt'
    result_list = ['didntLike', 'smallDoses', 'largeDoses']
    fly_miles = float(input('每周飞行的时长为:'))
    game_hours = float(input('每天玩游戏的时间为:'))
    ice_cream = float(input('每年吃的冰淇淋为:'))
    dating_mat, dating_labels = file_matrix(file_name)
    norm_mat, ranges, min_vals = auto_norm(dating_mat)
    input_array = np.array([fly_miles, game_hours, ice_cream])
    class_result = classify((input_array - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('可能喜欢这个人:', result_list[class_result - 1])

"""
将图像转换为测试向量
"""
def image_vector(file_name):
    return_vector=np.zeros((1,1024)) #返回一个1*1024的向量
    file=open(file_name)
    for i in range(32):
        line_string=file.readline() #每次读一行
        for j in range(32):
            return_vector[0,32*i+j]=int(line_string[j]) #这个向量本身也完全是由0,1构成，相当于将原来的矩阵每一行首尾相连
    return return_vector

"""
识别手写数字
"""
#识别手写数字
def handwritingClassTest():
    hwLabels = []
    training_file_list = os.listdir('./2022-06-08data/trainingDigits') #导入训练集，‘trainingDigits’是一个文件夹
    m = len(training_file_list) #计算训练样本个数
    training_mat = np.zeros((m,1024)) #初始化数据集，将所有训练数据用一个m行，1024列的矩阵表示
    for i in range(m):
        file_name_str = training_file_list[i]   #获得所有文件名，文件名格式‘x_y.txt’,x表示这个手写数字实际表示的数字（label）
        file_str = file_name_str.split('.')[0]  #去除 .txt
        class_num_str = int(file_str.split('_')[0])   #classnumber为每个样本的分类，用‘_’分割，取得label
        hwLabels.append(class_num_str)  #将所有标签都存进hwLables[]
        training_mat[i,:] = image_vector('./2022-06-08data/trainingDigits/%s' % file_name_str) #将文件转化为向量后存入trainingMat[],这里展现了灵活的文件操作
    test_file_list = os.listdir('./2022-06-08data/testDigits')  #迭代测试集
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]  #去除 .txt
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = image_vector('./2022-06-08data/testDigits/%s' % file_name_str)  #这部分针对测试集的预处理和前面基本相同
        classify_result = classify(vector_under_test, training_mat, hwLabels, 3)    #使用算法预测样本所属类别，调用了前面写的classify0()函数
        print ("分类结果:%d,真实类别:%d" % (classify_result, class_num_str))
        if (classify_result != class_num_str): error_count += 1.0       #算法结果与样本的实际分类做对比
    print ("\nthe total number of errors is: %d" % error_count)
    print ("\nthe total error rate is: %f" % (error_count/float(m_test)))

