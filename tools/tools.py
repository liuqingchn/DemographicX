#!/usr/bin/python
#-*- coding: utf:8 -*-
import pandas as pd
from setting import DATA_DIR
#from sklearn import neighbors
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import train_test_split

df = pd.read_csv(DATA_DIR+'/demographic_membership_training.csv')

#knn = neighbors.KNeighborsClassifier(n_neighbors = 2)

#clf = LogisticRegression()

#print knn

#clf.fit([[1],[2],[3],[4],[5],[6]], [0,0,0,1,1,1])
#print "**********",clf.predict([2])



class DemographicMembership():
    def predict(test_type,trainingData,testingData):
        pass



if __name__ == "__main__":
    #print sum(df['DEMO_X'])
    #print dir(df.columns)
    trainlist = []
    resultlist = []
    trainlist = np.array(trainlist)
    #clf_linear  = svm.SVC(kernel='linear').fit(trainlist, resultlist)
    for i in xrange(20):
        trainlist.append([ df[column][i] for column in df.columns if ((not isinstance(df[column][i], basestring)) and (column != "CONSUMER_ID"))])
        resultlist.append(df['DEMO_X'][i])
    #resultlist = np.zeros(resultlist)
    x_train, x_test, y_train, y_test = train_test_split(trainlist, resultlist, test_size = 0.2)

    print '**************&&&&&&&&&',len(x_train),len(y_train)
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')

    clf.fit(x_train, y_train)
    print trainlist[1]
    #print '-------------------------------',resultlist
    #knn.fit(trainlist,resultlist)
    #print "##########", clf_linear.predict(trainlist[10]), resultlist[10]
    print "ok"

