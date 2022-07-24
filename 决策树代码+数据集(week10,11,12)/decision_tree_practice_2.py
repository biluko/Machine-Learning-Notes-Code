import numpy as np
import pandas as pd
from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
path='./2022-06-19data/data_word.csv'
data=pd.read_csv(path)
print('数据集展示为:\n',data)
print(data.keys())

def cal_information_entropy(data):
    data_labels=data.iloc[:,-1]
    label_class=data_labels.value_counts() #标签计数
    entropy=0
    for key in label_class.keys():
        prob=label_class[key]/len(data_labels)
        entropy+=-prob*log(prob,2)
    return entropy
entropy=cal_information_entropy(data)
print('=='*30)
print('entropy=\n',entropy)
print('=='*30)

#计算给定数据属性a的信息增益
def cal_information_gain(data,a):
    entropy = cal_information_entropy(data)
    feature_class = data[a].value_counts()#特征有多少种可能
    #print(feature_class)
    #print('特征类型:\n', feature_class.keys())
    #print('==' * 30)
    gain_entropy = 0.0
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        entropy_v= cal_information_entropy(data.loc[data[a] == v])
        condition_entropy=weight*entropy_v #条件熵的每一项
        gain_entropy+=weight*entropy_v
        #print('权重值为:\n',weight)
        #print('对应的熵值为:\n',entropy_v)
        #print('对应的条件熵的子集为:\n',condition_entropy)
        #print(data.loc[data[a] == v],type(data.loc[data[a] == v]))
        #print('=='*30)
    infomation_entropy_gain = entropy - gain_entropy
    return infomation_entropy_gain
"""
print('色泽的信息增益为:\n',cal_information_gain(data,'色泽'))
print('==' * 30)
print('根蒂的信息增益为:\n',cal_information_gain(data,'根蒂'))
print('==' * 30)
print('敲声的信息增益为:\n',cal_information_gain(data,'敲声'))
print('==' * 30)
print('纹理的信息增益为:\n',cal_information_gain(data,'纹理'))
print('==' * 30)
print('脐部的信息增益为:\n',cal_information_gain(data,'脐部'))
print('==' * 30)
print('触感的信息增益为:\n',cal_information_gain(data,'触感'))
print('==' * 30)
"""

def cal_gain_ratio(data , a):
    #先计算固有值intrinsic_value
    IV_a = 0
    feature_class = data[a].value_counts()  # 特征有多少种可能
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        IV_a += -weight*log(weight,2)
    gain_ration = cal_information_gain(data,a)/IV_a
    return gain_ration

"""
#ID3算法，挑选信息增益最大的特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        info_gain = cal_information_gain(data,a)
        res[a] = info_gain
    res = sorted(res.items(),key=lambda x:x[1],reverse=True)
    return res[0][0]
print('获取分类的最佳特征:\n',get_best_feature(data))

"""
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = cal_information_gain(data, a)
        gain_ration = cal_gain_ratio(data,a)
        res[a] = (temp,gain_ration)
    res = sorted(res.items(),key=lambda x:x[1][0],reverse=True) #按信息增益排名
    res_avg = sum([x[1][0] for x in res])/len(res) #信息增益平均水平
    good_res = [x for x in res if x[1][0] >= res_avg] #选取信息增益高于平均水平的特征
    result =sorted(good_res,key=lambda x:x[1][1],reverse=True) #将信息增益高的特征按照增益率进行排名
    return result[0][0] #返回高信息增益中增益率最大的特征
print('获取最佳的划分属性:\n',get_best_feature(data))

#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]
print('获取标签最多的一类为:\n',get_most_label(data))

#将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
def drop_exist_feature(data, best_feature):
    attribute = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attribute]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
print(column_count)

"""
函数说明:创建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""
def create_tree(data):
    data_label = data.iloc[:,-1] #取分类标签
    if len(data_label.value_counts()) == 1: #只有一类
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): #所有数据的特征值一样，选样本最多的类作为分类结果
        return get_most_label(data)
    best_feature = get_best_feature(data) #根据信息增益得到的最优划分特征
    Tree = {best_feature:{}} #用字典形式存储决策树
    exist_vals = pd.unique(data[best_feature]) #当前数据下最佳特征的取值
    if len(exist_vals) != len(column_count[best_feature]): #如果特征的取值相比于原来的少了
        no_exist_attr = set(column_count[best_feature]) - set(exist_vals) #少的那些特征
        for no_feat in no_exist_attr:
            Tree[best_feature][no_feat] = get_most_label(data) #缺失的特征分类为当前类别最多的

    for item in drop_exist_feature(data,best_feature): #根据特征值的不同递归创建决策树
        Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree

my_tree=create_tree(data)
print('构造的决策树如下:\n',my_tree)

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
print(create_plot(my_tree))

def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0]
    second_dict = Tree[first_feature]
    input_first = test_data.get(first_feature)
    input_value = second_dict[input_first]
    if isinstance(input_value , dict): #判断分支还是不是字典
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label


test_data = {'色泽':'青绿','根蒂':'蜷缩','敲声':'浊响','纹理':'稍糊','脐部':'凹陷','触感':'硬滑'}
print('预测结果为:\n',predict(my_tree,test_data))