his is to use 'os' function
os.getcwd()     #This is to find out current working path (By default:C:\\Python34)
os.chdir('~\\Desktop\\Data')   #This is to change into Desktop

import scipy as sp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import logloss
from numpy import genfromtxt, savetxt

#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt('train.csv', delimiter=',', dtype='f8')[1:]    
target = np.array([x[0] for x in dataset])  #making it easier for tracking, useful for cv use
train = np.array([x[1:] for x in dataset])
test = genfromtxt('test.csv', delimiter=',', dtype='f8')[1:]
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
best_para =[400,2,1,18,320]


# free parameter  max_features /max_depth(about 18 is the best)/ max_leaf_nodes (no limit is the best)
#search for max_depth
for loop in range(15):
    result_buffer=[] 
    cv = cross_validation.KFold(len(train), n_folds=5)
    rf = RandomForestClassifier(n_estimators=best_para[0],
                                min_samples_split=best_para[1],
                                min_samples_leaf=best_para[2],
                                max_depth = best_para[3],
                                max_features = best_para[4]+loop*20
                                )    
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        result_buffer.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )
    ll=np.array(result_buffer).mean()
    if loop==0:
       min_ll=ll
       best_para_buffer= best_para[4]+loop*20
    if ll<min_ll:
       min_ll=ll
       best_para_buffer= best_para[4]+loop*20
    print(ll)
best_para[4]=best_para_buffer
print(best_para)

