import csv
import operator
import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
import matplotlib as mpl

def create_data():
    """
    # dataSet中最后一列记录的是类别，其余列记录的是特征值
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '1'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '1'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '1'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '1'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '1'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '1'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '1'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '1'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '0'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '0'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '0'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '0'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '0'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '0'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '0'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '0'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '0']
    ]
    # label中记录的是特征的名称
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    :return:
    """
    with open('./2022-06-19data/data_word.csv',encoding='utf-8',newline='\n') as f:
        reader=csv.reader(f) #此处读取到的数据是将每行数据当做列表返回的
        data=[]
        for row in reader: #此时输出的是一行行的列表
            data.append(row)
    labels=data[0][:-1]
    dataset=data[1:]
    return dataset, labels

def CART_Gini(dataset):
    # 计算参与训练的数据量，即训练集中共有多少条数据;
    num_examples = len(dataset)
    # 计算每个分类标签label在当前节点出现的次数，即每个类别在当前节点下别分别出现多少次，用作信息熵概率 p 的分子
    label_counts = {}
    # 每次读入一条样本数据
    for feature_vector in dataset:
        # 将当前实例的类别存储，即每一行数据的最后一个数据代表的是类别
        current_label = feature_vector[-1]
        # 为所有分类创建字典，每个键值对都记录了当前类别出现的次数
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 使用循环方法，依次求出公式求和部分每个类别所占的比例及相应的计算
    gini = 1  # 记录基尼值
    for key in label_counts:
        # 算出该类别在该节点数据集中所占的比例
        prob = label_counts[key] / num_examples
        gini -= prob * prob
    return gini

'''
1.data为该节点的父节点使用的数据集
2.retDataSet为某特征下去除特定属性后其余的数据
3.value为某个特征下的其中一个属性
4.index为当前操作的是哪一个特征
'''
def split_data(dataset, axis, value):
    # 存储划分好的数据集
    reduced_dataset = []
    for feature_vector in dataset:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            reduced_dataset.append(reduced_feature_vector)
    return reduced_dataset

def choose_best_feature_to_split(dataset):
    # 求该数据集中共有多少特征（由于最后一列为label标签，所以减1）
    num_features = len(dataset[0]) - 1
    # 初始化最优基尼指数，和最优的特征索引
    best_gini_index, best_feature = 1, -1
    # 依次遍历每一个特征，计算其基尼指数
    # 当前数据集中有几列就说明有几个特征
    for i in range(num_features):
        # 将数据集中每一个特征下的所有属性拿出来生成一个列表
        feature_list = [example[i] for example in dataset]
        # 使用集合对列表进行去重，获得去重后的集合，即：此时集合中每个属性只出现一次
        unique_values = set(feature_list)
        # 创建一个临时基尼指数
        new_gini_index = 0
        # 依次遍历该特征下的所有属性
        for value in unique_values:
            # 依据每一个属性划分数据集
            sub_dataset = split_data(dataset, i, value)  # 详情见splitDataSet函数
            # 计算该属性包含的数据占该特征包含数据的比例
            prob = len(sub_dataset) / len(dataset)
            # 计算每个属性结点的基尼值，并乘以相应的权重，再求他们的和，即为该特征节点的基尼指数
            new_gini_index += prob * CART_Gini(sub_dataset)
            # 比较所有特征中的基尼指数，返回最好特征划分的索引值，注意：基尼指数越小越好
        if (new_gini_index < best_gini_index):
            best_gini_index = new_gini_index
            best_feature = i
            # 返回最优的特征索引
    return best_feature

'''classList:为标签列的列表形式'''
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1  # classCount以字典形式记录每个类别出现的次数
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（好瓜/坏瓜），即出现次数最多的结果
    # sortedClassCount的形式是[(),(),(),...],每个键值对变为元组并以列表形式返回
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 返回的是出现类别次数最多的“类别”

def create_tree(dataset, labels):
    # classList中记录了该节点数据中“类别”一列，保存为列表形式
    class_list = [example[-1] for example in dataset]
    # 如果该结点为空集，则将其设为叶子结点，节点类型为其父节点中类别最多的类。
    if len(dataset) == 0:
        return majority_cnt(class_list)
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签
    # 如果数据集到最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，直接返回结果即可
    if class_list.count(class_list[0]) == len(class_list):  # count()函数是统计括号中的值在list中出现的次数
        return class_list[0]
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    # 如果最后只剩一个特征，那么出现相同label多的一类，作为结果
    if len(dataset[0]) == 1:  # 所有的特征都用完了，只剩下最后的标签列了
        return majority_cnt(class_list)
    # 选择最优的特征
    best_feature = choose_best_feature_to_split(dataset)  # 返回的是最优特征的索引
    # 获取最优特征
    best_feature_label = labels[best_feature]
    # 初始化决策树
    my_tree = {best_feature_label: {}}
    # 将使用过的特征数据删除
    del (labels[best_feature])
    # 在数据集中去除最优特征列，然后用最优特征的分支继续生成决策树
    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    # 遍历该特征下每个属性节点，继续生成决策树
    for value in unique_values:
        # 求出剩余的可用的特征
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data(dataset, best_feature, value), sub_labels)
    return my_tree

#绘图相关参数的设置
def plot_node(node_txt, center_pt, parent_pt, node_type):
    # annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    # nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
    # xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
    # bbox设置装注释盒子的样式,arrowprops设置箭头的样式
    '''
    figure points:表示坐标原点在图的左下角的数据点
    figure pixels:表示坐标原点在图的左下角的像素点
    figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
    其他位置是按相对图的宽高的比例取最小值
    axes points : 表示坐标原点在图中坐标的左下角的数据点
    axes pixels : 表示坐标原点在图中坐标的左下角的像素点
    axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
    '''
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
    num_leafs = 0 #初始化树的叶子节点个数
    # python3中my——tree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    first_str = list(my_tree.keys())[0]
    # 通过键名获取与之对应的值
    second_dict = my_tree[first_str]
    # 遍历树，secondDict.keys()获取所有的键
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            num_leafs += get_num_leafs(second_dict[key])
        else: #如果不是字典，则叶子结点的数目就加1
            num_leafs += 1
    return num_leafs #返回叶子节点的数目

"""
函数说明:获取决策树的层数
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def get_tree_depth(my_tree):
    max_depth = 0 #初始化决策树深度
    first_str = list(my_tree.keys())[0] #获取树的第一个键名
    second_dict = my_tree[first_str] #获取下一个字典,获取键名所对应的值
    for key in second_dict.keys(): #遍历树
        if type(second_dict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = get_tree_depth(second_dict[key]) + 1 #如果获取的键是字典，树的深度加1
        else:
            thisDepth = 1
        if thisDepth > max_depth: #去深度的最大值
            max_depth = thisDepth
    return max_depth #返回树的深度

"""
函数说明:标注有向边属性值
Parameters:
    cntr_pt、parent_pt - 用于计算标注位置
    txt_string - 标注的内容
Returns:
    无
"""
#绘制线中间的文字(0和1)的绘制
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0] #计算文字的x坐标
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1] #计算文字的y坐标
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
#设置画节点用的盒子的样式
leaf_node = dict(boxstyle="round4", fc="0.8")
decision_node = dict(boxstyle="sawtooth", fc="0.8")

#绘制树
def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree) #获取树的叶子节点
    depth = get_tree_depth(my_tree) #获取树的深度
    first_str = list(my_tree.keys())[0] #获取第一个键名
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off) #计算子节点的坐标
    plot_mid_text(cntr_pt, parent_pt, node_txt) #绘制线上的文字
    plot_node(first_str, cntr_pt, parent_pt, decision_node) #绘制节点
    second_dict = my_tree[first_str] #获取第一个键值
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d #计算节点y方向上的偏移量，根据树的深度
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key)) #递归绘制树
        else:
            # 更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            # 绘制非叶子节点
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            # 绘制箭头上的标志
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
    fig = plt.figure(1,figsize=(12,8),facecolor='white') #新建一个figure设置背景颜色为白色
    fig.clf() #清除figure
    axprops = dict(xticks=[], yticks=[])
    # 创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    # 的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree)) #获取树的叶子节点
    plot_tree.total_d = float(get_tree_depth(in_tree)) #获取树的深度
    #节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()

# 用上面训练好的决策树对新样本分类

def classify(my_tree,test_data):
    first_feature=list(my_tree.keys())[0] #获取根节点
    second_dict=my_tree[first_feature] #获取下一级分支
    #查找当前列表中第一个匹配first_feature变量的元素的索引
    input_first=test_data.get(first_feature)
    input_value=second_dict[input_first] #获取测试样本通过第一个特征分类器后的输出
    if isinstance(input_value,dict): #判断结点是否为字典来判断是否为根节点
        class_label=classify(input_value,test_data)
    else:
        class_label=input_value #如果到达叶子结点，则返回当前节点的分类标签
    return class_label

dataset, labels=create_data()
print('数据集为:\n',dataset)
print('数据标签为:\n',labels)
my_tree=create_tree(dataset,labels)
print(my_tree)
test_data_1 = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '平坦', '触感': '硬滑'}
result=classify(my_tree,test_data_1)
print('分类结果为'+'好瓜'if result=='1' else '分类结果为'+'坏瓜')
print(create_plot(my_tree))
