# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:49:50 2017

@author: wired_000
"""

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.min_samples_leaf = 5
clf.fit(iris.data, iris.target)
print (clf.predict([[5.1, 3.5, 1.4, 0.2]]))
print (clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("d:\iris.pdf") 