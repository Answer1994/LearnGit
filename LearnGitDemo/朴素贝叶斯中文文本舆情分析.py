# coding=utf-8
# /anaconda3/bin python

'''
Author: A君
Email: 13247352760@163.com
Wechat: 13247352760
date: 2019-07-12 09:49
desc:
'''

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

'''第一步：读取数据并分词'''
data = pd.read_csv('comment.csv')
print(data)
print(u"获取第一列内容")
col = data.iloc[:, 0]  # 取表中的第1列的所有值
arrs = col.values  # 取表中所有值
print(arrs)
stopwords = {}.fromkeys(['，', '。', '！', '这', '我', '非常'])  # 去除停用词

print(u"\n中文分词后结果")
corpus = []
for a in arrs:
    seglist = jieba.cut(a, cut_all=False)
    final = ''
    for seg in seglist:
        if seg not in stopwords:
            final = final + seg
    seg_list = jieba.cut(final, cut_all=False)
    output = ' '.join(list(seg_list))
    print(output)
    corpus.append(output)
print(corpus)

'''第二步：计算词频'''
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
X = vectorizer.fit_transform(corpus)
word = vectorizer.get_feature_names()
for w in word:
    print(w),
print('')
print(X.toarray())
tfidf = transformer.fit_transform(X)
weight = tfidf.toarray()
for i in range(len(weight)):
    print(u'------这里输出第', i, u'条文本的tf-idf权重------')
    for j in range(len(word)):
        print(word[j], weight[i][j])

'''第三步：数据分析'''
X = X.toarray()
x_train = X[:8]
x_test = X[8:]
y_train = [1, 1, 0, 0, 1, 0, 0, 1]
y_test = [1, 0]  # 1表示好评，0表示差评

clf = MultinomialNB().fit(x_train, y_train)
pre = clf.predict(x_test)
print(u'预测结果：', pre)
print(u'真是结果：', y_test)
print(classification_report(y_test, pre))
print(precision_recall_curve(y_test, pre))

'''因为数据量较小且不具备准确性，因此降维绘制图形，让实验结果尽可能的好'''
pca = PCA(n_components=2)
newData = pca.fit_transform(X)
print(newData)
Y = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
L1 = [n[0] for n in newData]
L2 = [n[1] for n in newData]
plt.scatter(L1, L2, s=200)
plt.show()
