from numpy import *
import re

def load_dataset():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return dataset,class_vector
dataset,class_vector=load_dataset()
print('数据集为:\n',dataset)
print('=='*30)
print('数据标签为:\n',class_vector)
print('=='*30)

def create_vocab_list(dataset):
    """创建一个包含所有文档且不出现重复词的列表"""
    vocab_set = set([])  #create empty set
    for document in dataset:
        vocab_set = vocab_set | set(document) #set()去掉列表中的重复词
    return list(vocab_set)

my_vacab_set=create_vocab_list(dataset)
print(my_vacab_set)

#词集模型
"""
输入为词汇表和文档，检查文档中的单词是否在词汇表中
采用词集模型:即对每条文档只记录某个词汇是否存在，而不记录出现的次数
创建一个与词汇表长度一致的0向量，在当前样本中出现的词汇标记为1
将一篇文档转换为词向量
"""
def set_of_words_vector(vocab_list, input_set):
    return_vector = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vector
print(set_of_words_vector(my_vacab_set,dataset[0]))
"""
朴素贝叶斯词袋模型
如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现中文档中所不能表达的某种信息
"""
def bag_word_vector(vocab_list,input_set):
    return_vector=[0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vector[vocab_list.index(word)]+=1
    return return_vector
print(bag_word_vector(my_vacab_set,dataset[0]))
"""
朴素贝叶斯分类器训练函数
"""
def train_native_bayes(train_matrix,train_category):
    num_train_docs=len(train_matrix)
    num_words=len(train_matrix[0])
    p=sum(train_category)/float(num_train_docs)
    p_0_num=ones(num_words)
    p_1_num=ones(num_words)
    p_0_denom=2.0
    p_1_denom=2.0
    for i in range(num_train_docs):
        if train_category[i]==1:
            p_1_num+=train_matrix[i]
            p_1_denom+=sum(train_matrix[i])
        else:
            p_0_num+=train_matrix[i]
            p_0_denom+=sum(train_matrix[i])
    p_1_vector=log(p_1_num/p_1_denom)
    p_0_vector=log(p_0_num/p_0_denom)
    return p_0_vector,p_1_vector,p

train_mat=[]
for i in dataset:
    train_mat.append(set_of_words_vector(my_vacab_set,i))
p_0_vector,p_1_vector,p=train_native_bayes(train_mat,class_vector)
print(p)
print(p_0_vector)

def classify_native_bayes(need_to_classify_vector, p_0_vector, p_1_vector, p_class):
    p_1 = sum(need_to_classify_vector * p_1_vector) + log(p_class)    #element-wise mult
    p_0 = sum(need_to_classify_vector * p_0_vector) + log(1.0 - p_class)
    if p_1 > p_0:
        return 1
    else:
        return 0

def testing_native_bayes():
    dataset,class_vector=load_dataset()
    my_vacab_set = create_vocab_list(dataset)
    my_vacab_set.sort()
    train_mat=[]
    for i in dataset:
        train_mat.append(set_of_words_vector(my_vacab_set, i))
    p_0_vector,p_1_vector,p = train_native_bayes(array(train_mat),array(class_vector))
    test_entry = ['love','my']
    this_doc = array(set_of_words_vector(my_vacab_set, test_entry))
    print(test_entry,'classified as: ',classify_native_bayes(this_doc,p_0_vector,p_1_vector,p))
    test_entry_1 = ['stupid','garbage']
    this_doc_1 = array(set_of_words_vector(my_vacab_set, test_entry_1))
    print(test_entry_1,'classified as: ',classify_native_bayes(this_doc_1,p_0_vector,p_1_vector,p))
print(testing_native_bayes())

"""
函数说明:接收一个大字符串并将其解析为字符串列表
"""
def text_parse(big_string):
    # 将字符串转换为字符列表
    list_of_tokens = re.split(r"[0-9!@#$%^&*()?\n~]",big_string) # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2] # 除了单个字母，例如大写的I，其它单词变成小写

"""
函数说明:测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
"""
def spam_test():
    doc_list=[]
    class_vector=[]
    full_text=[]
    for i in range(1,26): # 遍历25个txt文件
        word_list=text_parse(open('native_bayes  email dataset/spam/%d.txt'%i,'r').read()) # 读取每个垃圾邮件，并字符串转换成字符串列表
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_vector.append(1) # 标记垃圾邮件，1表示垃圾文件
        word_list=text_parse(open('native_bayes  email dataset/ham/%d.txt'%i,'r').read()) # 读取每个非垃圾邮件，并字符串转换成字符串列表
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_vector.append(0) # 标记正常邮件，0表示正常文件
    vocab_list=create_vocab_list(doc_list) # 创建词汇表，不重复
    training_set=list(range(50))
    test_set=[] # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(0,10): # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        rand_index=int(random.uniform(0,len(training_set))) # 随机选取索索引值
        test_set.append(training_set[rand_index]) # 添加测试集的索引值
        del (training_set[rand_index]) # 在训练集列表中删除添加到测试集的索引值
    train_mat=[]
    train_class=[] # 创建训练集矩阵和训练集类别标签系向量
    for doc_index in training_set: # 遍历训练集
        train_mat.append(set_of_words_vector(vocab_list,doc_list[doc_index])) # 将生成的词集模型添加到训练矩阵中
        train_class.append(class_vector[doc_index]) # 将类别添加到训练集类别标签系向量中
    p_0_vector,p_1_vector,p=train_native_bayes(array(train_mat),array(train_class)) # 训练朴素贝叶斯模型
    error_count=0 # 错误分类计数
    for doc_index in test_set: # 遍历测试集
        word_vector=set_of_words_vector(vocab_list,doc_list[doc_index]) # 测试集的词集模型
        if classify_native_bayes(array(word_vector),p_0_vector,p_1_vector,p)!=class_vector[doc_index]: # 如果分类错误
            error_count+=1 # 错误计数加1
            print('classify error:',doc_list[doc_index])
    print('the error rate is:',float(error_count)/len(test_set))

spam_test()

def one_cross_validate(train_set,train_class,test_set,test_class):
    #训练模型
    p_0_vector,p_1_vector,p_c_1 = train_native_bayes(array(train_set),array(train_class))
    error_count = 0
    #验证集进行测试
    for i in range(10):
        c = classify_native_bayes(array(test_set[i]),p_0_vector,p_1_vector,p_c_1)
        if c != test_class[i]:
            error_count += 1
    return error_count/10

def K_Cross_Validate(train_mat,train_class_vector):  #K折交叉验证 5
    rand_index = list(range(50))
    random.shuffle(rand_index)
    error_radio = 0.0
    for i in range(5):  #5次
        index = rand_index #随机索引
        #选取训练集、验证集索引
        train_set = []
        train_cls = []
        test_set = []
        test_cls = []
        test_set_index = set(rand_index[10*i:10*i+10])  # 测试集10
        train_set_index = set(index)-test_set_index  # 验证集
        #选取训练集、验证集数据
        for idx in train_set_index:
            train_set.append(train_mat[idx])
            train_cls.append(train_class_vector[idx])
        for idx in test_set_index:
            test_set.append(train_mat[idx])
            test_cls.append(train_class_vector[idx])
        print('第%d个子集的误差率为:'%(i+1),one_cross_validate(train_set,train_cls,test_set,test_cls))
        error_radio += one_cross_validate(train_set,train_cls,test_set,test_cls)
    return error_radio/5

def create_dataset():
    data_set_list=[]  #全部数据集
    class_vector = []    #标签值
    #获取数据
    spam_path = "native_bayes  email dataset/spam/{}.txt"  #获取文件路径
    ham_path = "native_bayes  email dataset/ham/{}.txt"
    for i in range(1, 26):  # 两个路径各有25个文件
        document_data_1 = open(spam_path.format(i), 'r').read()
        # 使用正则进行分割，除了空格、还有标点都可以用于分割
        word_vector = text_parse(document_data_1) # \W*表示匹配多个非字母、数字、下划线的字符
        data_set_list.append([item for item in word_vector if len(item) > 0])
        class_vector.append(1)
        document_data_2 = open(ham_path.format(i), 'r').read()
        # 使用正则进行分割，除了空格、还有标点都可以用于分割
        word_vector_2 = text_parse(document_data_2)  # \W*表示匹配多个非字母、数字、下划线的字符
        data_set_list.append([item for item in word_vector_2 if len(item) > 0])
        class_vector.append(0)
    return data_set_list, class_vector
data_set_list, class_vector=create_dataset()
vocab_list = create_vocab_list(data_set_list)
trainMulList = []
for doc in data_set_list:
    trainMulList.append(set_of_words_vector(vocab_list,doc))
print('=='*30)
print('5折交叉验证的错误率为:\n',K_Cross_Validate(trainMulList,class_vector))

