# coding=utf-8
# /anaconda3/bin python

'''
Author: A君
Email: 13247352760@163.com
Wechat: 13247352760
date: 2019-07-11 21:12
desc:
'''

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])

clf = GaussianNB()
clf.fit(X,Y)
pre = clf.predict(X)

print(u"数据集预测结果:", pre)
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y)) #增加一部分样本
print(clf_pf.predict([[-0.8, -1]]))