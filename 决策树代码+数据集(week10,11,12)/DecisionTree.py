from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=20)
estimator=DecisionTreeClassifier(criterion='entropy')
estimator.fit(x_train,y_train)
y_predict=estimator.predict(x_test)
score=estimator.score(x_test,y_test)
print('y的预测值为:\n',y_predict)
print('y的预测值与真实值:\n',y_predict==y_test)
print('准确率为:\n',score)
export_graphviz(estimator,out_file='./iris_tree.dot',feature_names=iris.feature_names)