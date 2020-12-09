from ensemble.sgh import SGH
from deslib.static.oracle import Oracle
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier, Perceptron
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils

X, y = utils.data_digits()
seed = 100000
base_learner = Perceptron()
kf = KFold(n_splits=5)
results = {'oracle_accuracy': [], 'oracle_std': [], 'ensemble_length': []}

"""
 - verifique quantas instâncias por classe foram incorretamente classificadas; 
 - verifique quantos hiperplanos por classe foram gerados
"""

pool_classifiers = SGH(base_estimator=base_learner)
scores = list()
fold = 1
for train_index, test_index in kf.split(X):
    print("Fold %d" % fold)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    pool_classifiers.fit(X_train, y_train)
    oracle = Oracle(pool_classifiers, random_state=seed)
    oracle.fit(X_train, y_train)
    
    
    score = oracle.score(X_test, y_test)
    scores.append(score)
    fold += 1

print("Acc (mean): %.2f" % np.mean(scores))
print("STD: %.2f" % np.std(scores))