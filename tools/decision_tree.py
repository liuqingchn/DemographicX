# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from tools import df


u''' 数据读入 '''
data  = []
labels = []
#with open("../Data/1.txt") as ifile:
    #for line in ifile:
        #tokens = line.strip().split(' ')
        #data.append([float(tk) for tk in tokens[:-1]])
        #labels.append(tokens[-1])
#x = np.array(data)
#labels = np.array(labels)
#y = np.zeros(labels.shape)
for i in xrange(len(df)):
    item = [ df[column][i] for column in df.columns if ((not isinstance(df[column][i], basestring)) and (column != "CONSUMER_ID") and (column != "DEMO_X"))]
    item = np.array(item)
    #print "$$$$$$$",np.isnan(item)
    if True in np.isnan(item):
        #print "have null value"
        pass
    else:
        data.append([ float(df[column][i]) for column in df.columns if ((not isinstance(df[column][i], basestring)) and (column != "CONSUMER_ID") and (column != "DEMO_X"))])
        labels.append(df['DEMO_X'][i])

#u''' 标签转换为0/1 '''
#y[labels=='fat']=1
print 'labels ==',labels

x = np.array(data)
#print "#####",x
labels = np.array(labels)
#y = np.zeros(labels.shape)
y = labels
#print "*******",y

u''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

u''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

u''' 把决策树结构写入文件 '''
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

u''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(max(clf.feature_importances_))

u'''测试结果的打印'''
answer = clf.predict(x_train)
#print(x_train)
print(answer)
print(y_train)
print(np.mean( answer == y_train))

u'''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict_proba(x)[:,1]
print(classification_report(y, answer, target_names = ['0', '1']))

