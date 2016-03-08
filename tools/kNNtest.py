__author__ = 'quantum'
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from tools import df

'''read data'''
data   = []
labels = []
#with open("../Data/1.txt") as ifile:
        #for line in ifile:
            #tokens = line.strip().split(' ')
            #data.append([float(tk) for tk in tokens[:-1]])
            #labels.append(tokens[-1])


for i in xrange(len(df)):
    item = [ df[column][i] for column in df.columns if ((not isinstance(df[column][i], basestring)) and (column != "CONSUMER_ID"))]
    item = np.array(item)
    #print "$$$$$$$",np.isnan(item)
    if True in np.isnan(item):
        print "have null value"
    else:
        data.append([ float(df[column][i]) for column in df.columns if ((not isinstance(df[column][i], basestring)) and (column != "CONSUMER_ID"))])
        labels.append(df['DEMO_X'][i])
print 'labels ==',labels

x = np.array(data)
print "#####",x
labels = np.array(labels)
#y = np.zeros(labels.shape)
y = labels
print "*******",y

''' change to 0/1 '''
#y[labels=='fat']=1
print "y==",y

print '-------------------------------'
print len(x)
print len(y)
print '--------------------------------'


''' diff test data and train data '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

''' draw pic '''
#h = .01
#x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
#y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 #                    np.arange(y_min, y_max, h))

''' train KNN '''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(x_train, y_train)

''' test and print'''
answer = clf.predict(x)
print "test and print"
#print(x)
print(answer)
print(y)
print(np.mean( answer == y))

''' precise and recall  '''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict_proba(x)[:,1]
#print(classification_report(y, answer, target_names = ['thin', 'fat']))


answer = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
z = answer.reshape(xx.shape)
plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

''' paint train data'''
#plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
#plt.xlabel('high')
#plt.ylabel('weight')
#plt.show()
