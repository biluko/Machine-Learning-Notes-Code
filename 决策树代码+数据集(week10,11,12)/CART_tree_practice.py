import csv
import operator
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
import matplotlib as mpl
file_name='./2022-06-19data/data_word.csv'
data=pd.read_csv(file_name)

def create_data():
    with open('./2022-06-19data/data_word.csv',encoding='utf-8',newline='\n') as f:
        reader=csv.reader(f) #此处读取到的数据是将每行数据当做列表返回的
        data=[]
        for row in reader: #此时输出的是一行行的列表
            data.append(row)
    features=data[0][:-1]
    dataset=data[1:]
    return dataset,features

dataset,features=create_data()
print('数据集为:\n',dataset)
print('=='*30)
print('数据标签为:\n',features,type(features))
print('=='*30)

def cal_gini_values(dataset):
    # 求总样本数
    num_examples=len(dataset)
    label_count={} #初始化一个字典用来保存每个标签出现的次数
    for feature_vector in dataset:
        current_label=feature_vector[-1] #逐个获取标签信息
        if current_label not in label_count.keys():
            #如果标签没有放入统计次数字典的话，就添加进去
            label_count[current_label]=0
        label_count[current_label]+=1
    gini=1.0
    for key in label_count:
        prob=float(label_count[key])/num_examples
        gini-=prob*prob
    return gini

gini=cal_gini_values(dataset)
print('基尼指数的值为:\n',gini)
print('=='*30)

# 提取子集合
# 功能：从dataSet中先找到所有第axis个标签值 = value的样本
# 然后将这些样本删去第axis个标签值，再全部提取出来成为一个新的样本集
"""
三个输入参数为：待划分的数据集、划分数据集的特征、需要返回的特征的值。
第4行，如果第axis个特征满足分类的条件，则进行以下操作：
第5行，featVec[:axis]是从0号元素开始取axis个元素，此时reducedFeatVec是前axis个元素，即0号到axis-1号元素；
第6行，featVec[axis+1:]是从axis+1号元素开始取直到最后一个。extend函数将两次取的元素拼接起来，即从原来的列表中去掉了axis号元素；
第7行，将去除元素后的列表再组合起来，成为一个新的列表，即满足第axis个特征的列表。
由此，完成了对第axis个特征的划分。
"""
def split_dataset(dataset,axis,value):
    sub_dataset=[]
    for feature_vector in dataset:
        if feature_vector[axis]==value:
            #下面两句将axis特征去掉，并将符合条件的添加到返回的数据集中
            reduced_feature_vector=feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            sub_dataset.append(reduced_feature_vector)
    return sub_dataset

"""
将当前样本集分割成特征i取值为value的一部分和取值不为value的一部分（二分）
"""
def split_dataset_2(dataset,axis,value):
    sub_dataset_1=[]
    sub_dataset_2=[]
    for feature_vector in dataset:
        if feature_vector[axis]==value:
            reduced_feature_vector=feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            sub_dataset_1.append(reduced_feature_vector)
        else:
            reduced_feature_vector=feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis+1:])
            sub_dataset_2.append(reduced_feature_vector)
    return sub_dataset_1,sub_dataset_2

def choose_best_feature_to_split(dataset):
    num_features=len(dataset[0])-1 #特征总数
    if num_features==1: #当只有一个特征时
        return 0
    #初始化最佳基尼系数
    best_gini_index=1
    # 初始化最优特征
    best_feature=-1
    #遍历所有特征，寻找最优特征和该特征下的最优切分点
    for i in range(num_features):
        feature_list=[example[i] for example in dataset]
        unique_vals=set(feature_list) #去重，每个属性值唯一
        new_gini_index=0
        # Gini字典中的每个值代表以该值对应的键作为切分点对当前集合进行划分后的Gini系数
        for value in unique_vals: # 对于当前特征的每个取值
            # 先求由该值进行划分得到的两个子集
            sub_dataset_1,sub_dataset_2=split_dataset_2(dataset,i,value)
            # 求两个子集占原集合的比例系数prob1 prob2
            prob_1=len(sub_dataset_1)/float(len(dataset))
            prob_2 =len(sub_dataset_2) / float(len(dataset))
            gini_index_1=cal_gini_values(sub_dataset_1)# 计算子集1的Gini系数
            gini_index_2=cal_gini_values(sub_dataset_2)# 计算子集2的Gini系数
            # 计算由当前最优切分点划分后的最终Gini系数
            new_gini_index=prob_1*gini_index_1+prob_2*gini_index_2
            # 更新最优特征和最优切分点
            if new_gini_index<best_gini_index:
               best_gini_index=new_gini_index
               best_feature=i
               best_split_point=value
    return best_feature,best_split_point

best_feature,best_split_point=choose_best_feature_to_split(dataset)
print(best_feature,best_split_point)
print('=='*30)

#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
# 初始化统计各标签次数的字典
# 键为各标签，对应的值为标签出现的次数
def majority_cnt(class_list):
    class_count={}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote]=0
        class_count[vote]+=1
    # 将classCount按值降序排列
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0] # 取sorted_labelCnt中第一个元素中的第一个值，即为所求

def create_tree(dataset,features):
    class_list=[example[-1] for example in dataset]# 求出训练集所有样本的标签
    # 先写两个递归结束的情况：
    # 若当前集合的所有样本标签相等（即样本已被分“纯”）
    # 则直接返回该标签值作为一个叶子节点
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
    # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
    # 则返回所含样本最多的标签作为结果
    if len(dataset[0])==1:
        return majority_cnt(class_list)
    # 下面是正式建树的过程
    # 选取进行分支的最佳特征的下标和最佳切分点
    best_feature,best_split_point=choose_best_feature_to_split(dataset)
    best_feature_label=features[best_feature]# 得到最佳特征
    my_tree={best_feature_label:{}}# 初始化决策树
    del(features[best_feature])# 使用过当前最佳特征后将其删去
    sub_labels=features[:]# 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    # 递归调用create_tree去生成新节点
    # 生成由最优切分点划分出来的二分子集
    sub_dataset_1,sub_dataset_2=split_dataset_2(dataset,best_feature,best_split_point)
    # 构造左子树
    my_tree[best_feature_label][best_split_point]=create_tree(sub_dataset_1,sub_labels)
    # 构造右子树
    my_tree[best_feature_label]['others']=create_tree(sub_dataset_2,sub_labels)
    return my_tree

my_tree=create_tree(dataset,features)
print(my_tree)
print('=='*30)

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

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

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

def classify(decision_tree,features,test_example):
    # 根节点代表的属性
    first_feature = list(decision_tree.keys())[0]
    # second_dict是第一个分类属性的值（也是字典）
    second_dict = decision_tree[first_feature]
    # 树根代表的属性，所在属性标签中的位置，即第几个属性
    index_of_first_feature = features.index(first_feature)
    # 对于second_dict中的每一个key
    for key in second_dict.keys():
        # 不等于'others'的key
        if key != 'others':
            if test_example[index_of_first_feature] == key:
            # 若当前second_dict的key的value是一个字典
                if type(second_dict[key]).__name__ == 'dict':
                    # 则需要递归查询
                    class_label = classify(second_dict[key], features,test_example)
                    # 若当前second_dict的key的value是一个单独的值
                else:
                    # 则就是要找的标签值
                    class_label = second_dict[key]
                # 如果测试样本在当前特征的取值不等于key，就说明它在当前特征的取值属于'others'
            else:
                # 如果second_dict['others']的值是个字符串，则直接输出
                if isinstance(second_dict['others'], str):
                    class_label = second_dict['others']
                    # 如果second_dict['others']的值是个字典，则递归查询
                else:
                    class_label = classify(second_dict['others'],features, test_example)
    return class_label

labels=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
test_example_1=['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
test_example_2=['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘']
result_1=classify(my_tree,labels,test_example_1)
result_2=classify(my_tree,labels,test_example_2)
print('分类结果为'+'好瓜'if result_1=='1' else '分类结果为'+'坏瓜')
print('分类结果为'+'好瓜'if result_2=='1' else '分类结果为'+'坏瓜')
print(create_plot(my_tree))