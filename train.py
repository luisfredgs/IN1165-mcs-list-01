import numpy as np
from deslib.static.oracle import Oracle
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from tqdm import tqdm
from sklearn.linear_model import Perceptron # base classifier
from sklearn.linear_model import SGDClassifier # base classifier
from ensemble.random_oracle import RandomOracleModel
from sklearn.model_selection import KFold
import pandas as pd
import utils


X, y = utils.data_digits()
seed = 100000
n_estimators = 10
pool_length = [10, 20, 30, 40, 50, 60, 80, 90, 100]
np.random.seed(seed)
base_learner = SGDClassifier(loss="perceptron", eta0=1.e-17,max_iter=200, learning_rate="constant", penalty=None)
pool_type = 'random_oracle_model'

print("Dataset size: %d" % X.shape[0])

kf = KFold(n_splits=5)
results = {'oracle_accuracy': [], 'oracle_std': [], 'ensemble_length': []}

for l in tqdm(pool_length):
    """
    With random Subspaces, estimators differentiate because of random subsets of the features.
    We can implement the random subspace ensemble using Bagging in scikit-learn, by keeping all training instances 
    but sampling features. According to (Aurlien, 2017, p. 188), we can achieve it by setting "bootstrap=False", "max_samples=1.0", 
    and "bootstrap_features=True"

    Reference: 
    Aurlien Gron. 2017. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques 
    to Build Intelligent Systems (1st. ed.). O'Reilly Media, Inc.
    """
    pool_types = {
        'bagging': BaggingClassifier(base_estimator=base_learner, n_estimators=l),
        'adaboost': AdaBoostClassifier(base_estimator=base_learner, n_estimators=l, algorithm='SAMME'),
        'random_subspace': BaggingClassifier(base_estimator=base_learner, n_estimators=l, bootstrap=False, 
                                            max_samples=1.0, max_features=0.50),
        'random_oracle_model': RandomOracleModel(base_estimator=base_learner, n_estimators=l)
    }

    pool_classifiers = pool_types[pool_type]
    scores = list()
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    

        pool_classifiers.fit(X_train, y_train)
        oracle = Oracle(pool_classifiers, random_state=seed)
        oracle.fit(X_train, y_train)
        
        if pool_type is 'random_subspace':
            score = oracle.score(X_test[np.random.permutation(len(X_test))][:, 0:int(0.50*X_test.shape[1])], y_test)
        elif pool_type is 'random_oracle_model':
            score = pool_classifiers.score(X_test, y_test)
        else:
            score = oracle.score(X_test, y_test)
        scores.append(score)
        
    
    results['oracle_accuracy'].append(np.mean(scores))
    results['oracle_std'].append(np.std(scores))
    results['ensemble_length'].append(l)    

df = pd.DataFrame(results)
df.to_csv("results/%s.csv" % pool_type)
print(df)