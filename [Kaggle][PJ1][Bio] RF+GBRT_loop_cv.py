import os       #This is to use 'os' function
os.getcwd()     #This is to find out current working path (By default:C:\\Python34)
os.chdir('~\\Desktop\\Data')   #This is to change into Desktop

import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
import logloss
from numpy import genfromtxt, savetxt

#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt('train.csv', delimiter=',', dtype='f8')[1:]    
target = np.array([x[0] for x in dataset])  #making it easier for tracking, useful for cv use
train = np.array([x[1:] for x in dataset])
test = genfromtxt('test.csv', delimiter=',', dtype='f8')[1:]

best_para_rt =[400,2,1,18,360]
best_para_grbt =[92,1,1,1,140]

for loop in range(11):       
       
        result_buffer=[]
        cv = cross_validation.KFold(len(train), n_folds=5)
        rf = RandomForestClassifier(n_estimators=best_para_rt[0],
                                min_samples_split=best_para_rt[1],
                                min_samples_leaf=best_para_rt[2],
                                max_depth = best_para_rt[3],
                                max_features = best_para_rt[4])
        
        gbrt = GradientBoostingClassifier(n_estimators=best_para_grbt[0],
                                          max_depth=best_para_grbt[1],                                      
                                          min_samples_split=best_para_grbt[2],
                                          min_samples_leaf=best_para_grbt[3],
                                          max_features=best_para_grbt[4],
                                          learning_rate=0.5, random_state=0)
        for traincv, testcv in cv:
            probas_rf   = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
            probas_gbrt = gbrt.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
            probas = 10/(loop+10)*probas_rf + loop/(loop+10)*probas_gbrt
            result_buffer.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )
            ll=np.array(result_buffer).mean()
            if loop==0: 
               min_ll=2 # just set a random number
               best=probas               
            if ll<min_ll:
               min_ll=ll
               best=probas
            print(ll)

predicted_probs = [[index + 1, x[1]] for index, x in enumerate(best)]
savetxt('result_grbt.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='MoleculeId,PredictedProbability', comments = '')
        
