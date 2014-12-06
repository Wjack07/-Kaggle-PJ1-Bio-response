his is to use 'os' function
os.getcwd()     #This is to find out current working path (By default:C:\\Python34)
os.chdir('~\\Desktop\\Data')   #This is to change into Desktop

import scipy as sp
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
import logloss
from numpy import genfromtxt, savetxt

#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt('train.csv', delimiter=',', dtype='f8')[1:]    
target = np.array([x[0] for x in dataset])  #making it easier for tracking, useful for cv use
train = np.array([x[1:] for x in dataset])
test = genfromtxt('test.csv', delimiter=',', dtype='f8')[1:]

best_para =[92,1,1,1,140]
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

#search for n_estimate

for loop in range(20):                                       # from 0 to 2
    result_buffer=[] 
    cv = cross_validation.KFold(len(train), n_folds=5)
    gbrt = GradientBoostingClassifier(n_estimators=best_para[0],
                                          max_depth=best_para[1],                                      
                                          min_samples_split=best_para[2],
                                          min_samples_leaf=best_para[3],
                                          max_features=best_para[4]+loop*10,
                                          learning_rate=0.5, random_state=0)    # cannot be 0
    for traincv, testcv in cv:
        probas = gbrt.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        result_buffer.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )
    ll=np.array(result_buffer).mean()
    if loop==0:
        min_ll=ll
        best_para_buffer=best_para[4]+loop*10
    if ll<min_ll:
        min_ll=ll
        best_para_buffer=best_para[4]+loop*10
    print(ll)
best_para[4]=best_para_buffer
print(best_para,' gbrt')

#max_features=best_para[5],
#max_leaf_nodes=best_para[2],
