import pandas as pd
import numpy as np
from math import log
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
path='./2022-06-19data/data_word.csv'
data=pd.read_csv(path)
print('数据集展示为:\n',data)
print(data.keys())
print('=='*30)

#计算基尼系数
def gini(data):
    data_label=data.iloc[:,-1]
    label_num = data_label.value_counts()  # 有几类，每一类的数量
    res=0
    for k in label_num.keys():
        p_k=label_num[k]/len(data_label)
        res+=p_k**2
    gini_value=1-res
    return gini_value
gini_value=gini(data)
print('数据集的纯度—基尼值G(D)为:\n',gini_value)
print('=='*30)

# 计算每个特征取值的基尼指数，找出最优切分点
def gini_index(data,a):
    feature_class=data[a].value_counts()
    res=[]
    for v in feature_class.keys():
        weight=feature_class[v]/data.shape[0]
        gini_value=gini(data.loc[data[a]==v])
        #res+=weight*gini_value
        res.append([v,weight*gini_value])
    res=sorted(res,key=lambda x:x[-1])
    return res[0]

for a in ['色泽','根蒂','敲声','纹理','脐部','触感']:
    res=gini_index(data,a)
    print('%s的基尼指数为:\n'%a, res)
    print('=='*30)

#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#挑选最优特征，即基尼指数最小的特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = gini_index(data, a)  #temp是列表，【feature_value, gini】
        res[a] = temp
    res = sorted(res.items(),key=lambda x:x[1][1]) #按照res[1][1]进行排序
    return res[0][0], res[0][1][0]
print('获取最佳划分特征:\n',get_best_feature(data))
print('=='*30)

def drop_exist_feature(data, best_feature, value, type):
    attr = pd.unique(data[best_feature]) #表示特征所有取值的数组
    if type == 1: #使用特征==value的值进行划分
        new_data = [[value], data.loc[data[best_feature] == value]]
    else:
        new_data = [attr, data.loc[data[best_feature] != value]]
    new_data[1] = new_data[1].drop([best_feature], axis=1) #删除该特征
    return new_data

#创建决策树
def get_most_label(label_list):
    return label_list.value_counts().idxmax()

# 创建决策树，传入的是一个dataframe，最后一列为label
def create_tree(data):
    feature = data.columns[:-1]
    label_list = data.iloc[:, -1]
    #如果样本全属于同一类别C，将此节点标记为C类叶节点
    if len(pd.unique(label_list)) == 1:
        return label_list.values[0]
    #如果待划分的属性集A为空，或者样本在属性A上取值相同，则把该节点作为叶节点，并标记为样本数最多的分类
    elif len(feature)==0 or len(data.loc[:,feature].drop_duplicates())==1:
        return get_most_label(label_list)
    #从A中选择最优划分属性
    best_attr,lable = get_best_feature(data)
    tree = {best_attr: {}}
    #对于最优划分属性的每个属性值，生成一个分支
    for attr,gb_data in data.groupby(by=best_attr):
        if len(gb_data) == 0:
            tree[best_attr][attr] = get_most_label(label_list)
        else:
            #在data中去掉已划分的属性
            new_data = gb_data.drop(best_attr,axis=1)
            #递归构造决策树
            tree[best_attr][attr] = create_tree(new_data)
    return tree
my_tree=create_tree(data)
print(my_tree)

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

