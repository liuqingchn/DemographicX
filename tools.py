#!/usr/bin/python
#-*- coding: utf:8 -*-
import pandas as pd
df = pd.read_csv('demographic_membership_training.csv')
print sum(df['DEMO_X'])
print '####',len(dir(df))
#for item in dir(df):
#	print k,v
    #print item
print dir(df)
print df.columns
print len(df)


class DemographicMembership():
	def predict(test_type,trainingData,testingData):
		pass



if __name__ == "__main__":
    print "ok"

