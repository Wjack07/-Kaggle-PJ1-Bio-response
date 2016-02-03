by ak321/Wjack07 @ 5Dec2014 
---
This work has been following this link:
https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience

The idea is from them, and also many parts of the sources code.
However, to pursue a better performance, I modify the code into three parts.
*I choose two classifiers: Random Forest & Gradient boosting

1. Optimizing parameter for Random Forest:
In the rf.py file, the iteration concept to look for a better set of parameter is shown.
There is one sub iteration, but it can be extended into several parameter.
And then use another bigger iteration to subsribe it (so to run through the searching again.)

2. Optimizing Gradient boosting
The same concept, just modified for Gradient boosting
The gb.py code present the bigger loop concept to search optimized parameters

3. Mixing
Simply linearly mixing two classifier with different weighting.
The weighting is searched by the optimization concept above.

Note:
*The logloss and cross-validation has been included into the code, so things can be done all in one.
*The optimization concept itself is not optimized, but it provides the concept
*Most classifiers should be considered.

The optimized parameters for random forest is:
(n_estimators=400,min_samples_split=2,min_samples_leaf=1,max_depth=18,max_features=360]
The performance is about 0.431

The optimized parameters for gradient boosting is:
(n_estimators=92,max_depth=1,min_samples_split=1,min_samples_leaf=1,max_features=140])
The performance is about 0.499

Unexpectedly, the best combination iteration shows the performance is quite uneven,
and there is no a clear trendn.
The best ratio is surprisingly rf:grbt=1:0  <= so linear mixing between rf and gbrt is useless

---
6Dec2014
Surprisingly, the local cross validation score is about 0.43, and the uploading performance is 0.398 (around 200 out from 6xx participants, not too bad.)
