import pandas as pd
from numpy import *
import copy
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

# 计算数据集的基尼指数
def cal_gini(dataset):
    num_examples = len(dataset)
    label_counts = {}
    # 给所有可能分类创建字典
    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    gini = 1.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_examples
        gini -= prob * prob
    return gini

# 对离散变量划分数据集，取出该特征取值为value的所有样本
def split_dataset(dataset, axis, value):
    reduced_dataset = []
    for feature_vector in dataset:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            reduced_dataset.append(reduced_feature_vector)
    return reduced_dataset

# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def split_continuous_dataset(dataset, axis, value, direction):
    reduced_dataset = []
    for feature_vector in dataset:
        if direction == 0:
            if feature_vector[axis] > value:
                reduced_feature_vector = feature_vector[:axis]
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                reduced_dataset.append(reduced_feature_vector)
        else:
            if feature_vector[axis] <= value:
                reduced_feature_vector = feature_vector[:axis]
                reduced_feature_vector.extend(feature_vector[axis + 1:])
                reduced_dataset.append(reduced_feature_vector)
    return reduced_dataset

# 选择最好的数据集划分方式
def choose_best_feature_to_split(dataset, labels):
    num_features = len(dataset[0]) - 1
    best_gini_index = 1
    best_feature = -1
    best_split_dict = {}
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        # 对连续型特征进行处理
        if type(feature_list[0]).__name__ == 'float' or type(feature_list[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sorted_feature_list = sorted(feature_list)
            split_list = []
            for j in range(len(sorted_feature_list) - 1):
                split_list.append((sorted_feature_list[j] + sorted_feature_list[j + 1]) / 2.0)
            best_split_gini = 10000
            split_len = len(split_list)
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(split_len):
                value = split_list[j]
                new_gini_index = 0.0
                sub_dataset_0 = split_continuous_dataset(dataset, i, value, 0)
                sub_dataset_1 = split_continuous_dataset(dataset, i, value, 1)
                prob0 = len(sub_dataset_0) / float(len(dataset))
                new_gini_index += prob0 * cal_gini(sub_dataset_0)
                prob1 = len(sub_dataset_1) / float(len(dataset))
                new_gini_index += prob1 * cal_gini(sub_dataset_1)
                if new_gini_index < best_split_gini:
                    best_split_gini = new_gini_index
                    best_split = j
            # 用字典记录当前特征的最佳划分点
            best_split_dict[labels[i]] = split_list[best_split]
            gini_index = best_split_gini
        # 对离散型特征进行处理
        else:
            unique_values = set(feature_list)
            new_gini_index = 0.0
            # 计算该特征下每种划分的信息熵
            for value in unique_values:
                sub_dataset = split_dataset(dataset, i, value)
                prob = len(sub_dataset) / float(len(dataset))
                new_gini_index += prob * cal_gini(sub_dataset)
            gini_index = new_gini_index
        if gini_index < best_gini_index:
            best_gini_index = gini_index
            best_feature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    # 并将特征名改为 name<=value的格式
    if type(dataset[0][best_feature]).__name__ == 'float' or type(dataset[0][best_feature]).__name__ == 'int':
        best_split_value = best_split_dict[labels[best_feature]]
        labels[best_feature] = labels[best_feature] + '<=' + str(best_split_value)
        for i in range(shape(dataset)[0]):
            if dataset[i][best_feature] <= best_split_value:
                dataset[i][best_feature] = 1
            else:
                dataset[i][best_feature] = 0
    return best_feature

# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    return max(class_count)

def create_tree(dataset, labels,data_full,labels_full):
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
    best_feature = choose_best_feature_to_split(dataset,labels)  # 返回的是最优特征的索引
    # 获取最优特征
    best_feature_label = labels[best_feature]
    # 初始化决策树
    my_tree = {best_feature_label: {}}
    # 在数据集中去除最优特征列，然后用最优特征的分支继续生成决策树
    feature_values = [example[best_feature] for example in dataset]
    unique_values = set(feature_values)
    if type(dataset[0][best_feature]).__name__=='str':
        current_label=labels_full.index(labels[best_feature])
        feature_values_full=[example[current_label] for example in data_full]
        unique_values_full=set(feature_values_full)
    # 将使用过的特征数据删除
    del (labels[best_feature])
    # 遍历该特征下每个属性节点，继续生成决策树
    for value in unique_values:
        # 求出剩余的可用的特征
        sub_labels = labels[:]
        if type(dataset[0][best_feature]).__name__=='str':
            unique_values_full.remove(value)
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels,data_full,labels_full)
    if type(dataset[0][best_feature]).__name__=='str':
        for value in unique_values_full:
            my_tree[best_feature_label][value]=majority_cnt(class_list)
    return my_tree

# 由于在Tree中，连续值特征的名称以及改为了  feature<=value的形式
# 因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
def classify(input_tree, feature_labels, test_vector):
    first_feature = list(input_tree.keys())[0]
    if '<=' in first_feature:
        feature_value = float(re.compile("(<=.+)").search(first_feature).group()[1:])
        feature_key = re.compile("(.+<=)").search(first_feature).group()[:-1]
        second_dict = input_tree[first_feature]
        feature_index = feature_labels.index(feature_key)
        if test_vector[feature_index] <= feature_value:
            judge = 1
        else:
            judge = 0
        for key in second_dict.keys():
            if judge == int(key):
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = classify(second_dict[key], feature_labels, test_vector)
                else:
                    class_label = second_dict[key]
    else:
        second_dict = input_tree[first_feature]
        feature_index = feature_labels.index(first_feature)
        for key in second_dict.keys():
            if test_vector[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = classify(second_dict[key], feature_labels, test_vector)
                else:
                    class_label = second_dict[key]
    return class_label

# 测试决策树正确率
def testing(my_tree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(my_tree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    # print 'myTree %d' %error
    return float(error)

# 测试投票节点正确率
def testing_major(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    # print 'major %d' %error
    return float(error)

# 后剪枝
def post_pruning_tree(input_tree, dataset, data_test,labels):
    first_feature = list(input_tree.keys())[0]  # 根结点

    second_dict = input_tree[first_feature] #获取下一分支
    class_list = [example[-1] for example in dataset]
    feature_key = copy.deepcopy(first_feature) #此特征值
    if '<=' in first_feature:
        feature_value = float(re.compile("(<=.+)").search(first_feature).group()[1:])
        feature_key = re.compile("(.+<=)").search(first_feature).group()[:-1]
    label_index = labels.index(feature_key)
    temp_labels = copy.deepcopy(labels)
    del (labels[label_index])
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            if type(dataset[0][label_index]).__name__ == 'str':
                input_tree[first_feature][key] = post_pruning_tree(second_dict[key], \
                                                           split_dataset(dataset, label_index, key),
                                                           split_dataset(data_test, label_index, key),
                                                           copy.deepcopy(labels))
            else:
                input_tree[first_feature][key] = post_pruning_tree(second_dict[key], \
                                                           split_continuous_dataset(dataset, label_index, feature_value, key), \
                                                           split_continuous_dataset(data_test, label_index, feature_value,
                                                                                  key), \
                                                           copy.deepcopy(labels))
    if testing(input_tree, data_test, temp_labels) <= testing_major(majority_cnt(class_list), data_test):
        return input_tree
    return majority_cnt(class_list)

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
    first_str = list(my_tree.keys())[0] #python3中my——tree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    second_dict = my_tree[first_str] #获取下一组字典
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

def create_dataset():
    df = pd.read_csv('./2022-06-19data/train_data.csv')
    df2=pd.read_csv('./2022-06-19data/test_data.csv')
    data = df.values[0:,:].tolist()
    data_full = data[:]
    data_test = df2.values[0:,:].tolist()
    labels = df.columns.values[0:-1].tolist()
    labels_full = labels[:]
    return data,labels,data_full,labels_full,data_test

data,labels,data_full,labels_full,data_test=create_dataset()
print(data)
print(data_test)
print(labels)
print(labels_full)
print(labels==labels_full)
my_tree=create_tree(data,labels,data_full,labels_full)
print(my_tree)
my_tree_2=post_pruning_tree(my_tree,data,data_test,labels_full)
print(create_plot(my_tree))
print(create_plot(my_tree_2))
