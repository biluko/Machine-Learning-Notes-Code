from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
iris=load_iris()#获取数据
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=6)#数据集划分
transfer=StandardScaler()#转化器类
x_train=transfer.fit_transform(x_train)#对训练数据进行标准化，y是种类，不需要标准化
x_test=transfer.transform(x_test)#对测试数据的转换
estimator=KNeighborsClassifier()#实现KNN算法的估计器
param_dict={'n_neighbors':[1,3,5]}
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=5)
estimator.fit(x_train,y_train)
y_predict=estimator.predict(x_test)
print('target_names:')
print('0:setosa   1:versicolor   2:virginica')
print('y_predict:\n',y_predict)
print('比较真实值和预测值:\n',y_predict==y_test)
score=estimator.score(x_test,y_test)
print('准确率为:\n',score)
print('最佳参数:\n',estimator.best_params_)
print('最佳结果:\n',estimator.best_score_)