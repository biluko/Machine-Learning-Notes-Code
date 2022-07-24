# 导入包
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('./2022-06-19data/data_word.csv')
print(df)
#将特征值化为数字
df['色泽']=df['色泽'].map({'浅白':1,'青绿':2,'乌黑':3})
df['根蒂']=df['根蒂'].map({'稍蜷':1,'蜷缩':2,'硬挺':3})
df['敲声']=df['敲声'].map({'清脆':1,'浊响':2,'沉闷':3})
df['纹理']=df['纹理'].map({'清晰':1,'稍糊':2,'模糊':3})
df['脐部']=df['脐部'].map({'平坦':1,'稍凹':2,'凹陷':3})
df['触感'] = df['触感'].map({'硬滑':1,'软粘':2})
x_train,x_test,y_train,y_test=train_test_split(df[['色泽','根蒂','敲声','纹理','脐部','触感']],df['好瓜'],test_size=0.2)
gini=tree.DecisionTreeClassifier(random_state=0,max_depth=5)
gini.fit(x_train,y_train)
pred = gini.predict(x_test)
print('模型的预测准确率为:\n',accuracy_score(y_test,pred))
labels = ['se ze', 'gen di', 'qiao sheng', 'wen li', 'qi bu', 'chugan']
model=DecisionTreeClassifier(random_state=0)
path=model.cost_complexity_pruning_path(x_train,y_train)
plt.plot(path.ccp_alphas, path.impurities, marker='o', drawstyle='steps-post')
plt.xlabel('alpha (cost-complexity parameter)')
plt.ylabel('Total Leaf Impurites')
plt.title('Total Leaf Impurites vs alpha for Training Set')
print("模型的复杂度参数与不纯度：", max(path.ccp_alphas), max(path.impurities))

param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=3,shuffle=True, random_state=1)
model_0 = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, cv=kfold)
model_0.fit(x_train, y_train)
print("最优参数：", model_0.best_params_)
model_1 = model_0.best_estimator_
print("预测准确率：", model_1.score(x_test, y_test))
plot_tree(model_1, feature_names=labels, node_ids=True,proportion=True, rounded=True, precision=2)
plt.tight_layout()
plt.show()