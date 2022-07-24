from math import log
import operator
import matplotlib.pyplot as plt
import pandas as pd
#计算给定数据的香农熵
def calculate_entropy(dataset):
    num_entropy=len(dataset) #获取数据集样本个数
    num_labels={} #初始化一个字典用来保存每个标签出现的次数
    for feature_vector in dataset:
        current_label=feature_vector[-1] #逐个获取标签信息
        if current_label not in num_labels.keys(): # 如果标签没有放入统计次数字典的话，就添加进去
            num_labels[current_label]=0
        num_labels[current_label]+=1
    entropy=0.0 #初始化香农熵
    for key in num_labels:
        prob=float(num_labels[key])/num_entropy #选择该标签的概率
        entropy-=prob*log(prob,2) #公式计算
    return entropy

def create_dataset():
    dataset=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def split_dataset(dataset,axis,value):
    new_dataset=[] #创建新列表以存放满足要求的样本
    for feature_vector in dataset:
        if feature_vector[axis]==value:
            #下面这两句用来将axis特征去掉，并将符合条件的添加到返回的数据集中
            reduced_feature_vector=feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            new_dataset.append(reduced_feature_vector)
    return new_dataset

"""
ID3
"""
def choose_best_feature_to_split(dataset):
    number_features=len(dataset[0])-1 #获取样本集中特征个数，-1是因为最后一列是label
    base_entropy = calculate_entropy(dataset)  # 计算根节点的信息熵
    best_info_gain = 0.0  # 初始化信息增益
    best_feature = -1  # 初始化最优特征的索引值
    for i in range(number_features):  # 遍历所有特征，i表示第几个特征
        #将dataset中的数据按行依次放入example中，然后取得example中的example[i]元素，即获得特征i的所有取值
        feature_list = [example[i] for example in dataset]
        #由上一步得到了特征i的取值，比如[1,1,1,0,0]，使用集合这个数据类型删除多余重复的取值，则剩下[1,0]
        unique_vals = set(feature_list) # 获取无重复的属性特征值
        new_entropy=0.0
        for value in unique_vals:
            sub_dataset =split_dataset(dataset, i, value) #逐个划分数据集，得到基于特征i和对应的取值划分后的子集
            prob = len(sub_dataset)/float(len(dataset)) #根据特征i可能取值划分出来的子集的概率
            new_entropy += prob * calculate_entropy(sub_dataset) #求解分支节点的信息熵
        info_gain = base_entropy - new_entropy  #计算信息增益，g(D,A)=H(D)-H(D|A)
        if (info_gain > best_info_gain):  # 对循环求得的信息增益进行大小比较
            best_info_gain=info_gain
            best_feature = i #如果计算所得信息增益最大，则求得最佳划分方法
    return best_feature #返回划分属性（特征）

"""
C4.5
"""
def choose_best_feature_to_split_2(dataset):
    """
    按照最大信息增益比划分数据
    :param dataset: 样本数据，如： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    :return:
    """
    number_feature = len(dataset[0]) - 1 # 特征个数，如：不浮出水面是否可以生存	和是否有脚蹼
    base_entropy = calculate_entropy(dataset) # 经验熵H(D)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(number_feature):
        feature_list = [number[i] for number in dataset]  # 得到某个特征下所有值（某列）
        unique_feature_list = set(feature_list)  # 获取无重复的属性特征值
        new_entropy = 0
        split_info = 0.0
        for value in unique_feature_list:
            sub_dataset = split_dataset(dataset,i, value)
            prob = len(sub_dataset) / float(len(dataset))  # 即p(t)
            new_entropy += prob * calculate_entropy(sub_dataset)  # 对各子集香农熵求和
            split_info += -prob * log(prob, 2)
        info_gain = base_entropy - new_entropy  # 计算信息增益，g(D,A)=H(D)-H(D|A)
        if split_info == 0:  # fix the overflow bug
            continue
        info_gain = info_gain / split_info
        # 最大信息增益比
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    #分解为元组列表，operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是逆序，即按照从大到小的顺序排列
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_decision_tree(dataset,labels):
    class_list = [example[-1] for example in dataset] #获取类别标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0] #类别完全相同则停止继续划分
    if len(dataset[0]) == 1:
        return majority_cnt(class_list) #遍历完所有特征时返回出现次数最多的类别
    best_feature = choose_best_feature_to_split(dataset) #选取最优划分特征
    best_feature_label = labels[best_feature] #获取最优划分特征对应的属性标签
    my_tree = {best_feature_label:{}} #存储树的所有信息
    del(labels[best_feature]) #删除已经使用过的属性标签
    feature_values = [example[best_feature] for example in dataset] #得到训练集中所有最优特征的属性值
    unique_vals = set(feature_values) #去掉重复的属性值
    for value in unique_vals: #遍历特征，创建决策树
        sub_labels = labels[:] #剩余的属性标签列表
        my_tree[best_feature_label][value] = create_decision_tree(split_dataset(dataset, best_feature, value),sub_labels) #递归函数实现决策树的构建
    return my_tree

def classify(input_tree,feature_labels,test_vector):
    first_string = list(input_tree.keys())[0] #获取根节点
    second_dict = input_tree[first_string] #获取下一级分支
    feature_index = feature_labels.index(first_string) #查找当前列表中第一个匹配firstStr变量的元素的索引
    key = test_vector[feature_index] #获取测试样本中，与根节点特征对应的取值
    value_feature = second_dict[key]#获取测试样本通过第一个特征分类器后的输出
    if isinstance(value_feature, dict): #判断节点是否为字典来以此判断是否为叶节点
        class_label = classify(value_feature, feature_labels, test_vector)
    else:
        class_label = value_feature#如果到达叶子节点，则返回当前节点的分类标签
    return class_label

"""
函数说明:绘制结点
Parameters:
    node_txt - 结点名
    center_pt - 文本位置
    parent_pt - 标注的箭头位置
    node_type - 结点格式
Returns:
    无
"""
def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-") #定义箭头格式
    # 绘制结点
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',xytext=center_pt, textcoords='axes fraction',va="center", ha="center", bbox=node_type, arrowprops=arrow_args)
"""
函数说明:获取决策树叶子结点的数目
Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""
def get_num_leafs(my_tree):
    num_leafs = 0 #初始化叶子
    first_str = list(my_tree.keys())[0] ##python3中my——tree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    second_dict = my_tree[first_str] ##获取下一组字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

"""
函数说明:获取决策树的层数
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def get_tree_depth(my_tree):
    max_depth = 0 #初始化决策树深度
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str] #获取下一个字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = get_tree_depth(second_dict[key]) + 1
        else:
            thisDepth = 1
        if thisDepth > max_depth: #更新层数
            max_depth = thisDepth
    return max_depth

"""
函数说明:标注有向边属性值
Parameters:
    cntr_pt、parent_pt - 用于计算标注位置
    txt_string - 标注的内容
Returns:
    无
"""
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)

"""
函数说明:绘制决策树
Parameters:
    my_tree - 决策树(字典)
    parent_pt - 标注的内容
    node_txt - 结点名
Returns:
    无
"""

leaf_node = dict(boxstyle="round4", fc="0.8")
decision_node = dict(boxstyle="sawtooth", fc="0.8")

def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d

"""
函数说明:创建绘制面板
Parameters:
    in_tree - 决策树(字典)
Returns:
    无
"""
def create_plot(in_tree):
    fig = plt.figure(1,figsize=(12,8),facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()

"""
函数说明:存储决策树
Parameters:
    inputTree - 已经生成的决策树
    filename - 决策树的存储文件名
Returns:
    无
"""
def store_tree(input_tree,file_name):
    import pickle
    with open(file_name,'wb+') as fw:
        pickle.dump(input_tree,fw)

"""
函数说明:读取决策树
Parameters:
    file_name - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
"""
def grab_tree(file_name):
    import pickle
    fr = open(file_name, 'rb')
    return pickle.load(fr)
if __name__=='__main__':
    """
    my_dataset,labels=create_dataset()
    print('=='*30)
    print('my_dataset=',my_dataset)
    print('labels=',labels)
    entropy = calculate_entropy(my_dataset)
    print('entropy=', entropy)
    print(type(my_dataset))
    print('=='*30)
    a=[1,1,0]
    b=a[:0]
    print('b=',b)
    c=a[:2]
    print('c=',c)
    c.extend(a[3:])
    print('c.extend(a[3:])=',c.extend(a[3:]))
    print('=='*30)
    split_dataset = split_dataset(my_dataset, 0, 0)
    print('split_dataset=', split_dataset)
    print('=='*30)
    print('=='*30)
    selection_1=choose_best_feature_to_split(my_dataset)
    print('selection_1=',selection_1)
    print('=='*30)
    selection_2=choose_best_feature_to_split_2(my_dataset)
    print('selection_2=',selection_2)
    print('=='*30)
    my_tree=create_decision_tree(my_dataset,labels)
    print('my_tree=',my_tree)
    print('=='*30)
    my_dataset, labels = create_dataset()
    result=classify(my_tree,labels,[1,0])
    print('result=',result)
    create_plot(my_tree)
    print('=='*30)
    path='lenses.txt'
    data=pd.read_csv(path,header=None)
    print(data)
    fr=open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print('=='*30)
    print('lenses=',lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = create_decision_tree(lenses, lensesLabels)
    print('=='*30)
    create_plot(lenses_tree)
    store_tree(my_tree,'decision_tree_practice.txt')
    myTree = grab_tree('decision_tree_practice.txt')
    print(myTree)
    """
