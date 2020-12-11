from sklearn.ensemble import BaseEnsemble
import numpy as np
from scipy import stats

class RandomOracleModel(BaseEnsemble):
    
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        self.estimators_ = []
        
            
    def fit(self, X, y):
        N = X.shape[0]
        self.E = {f'D_{i}':{'left:':None, 'right':None} for i in range(self.n_estimators)}
        self.H = {f'H_{i}':None for i in range(self.n_estimators)}
        
        for i in range(self.n_estimators):     
            
            while True:
                try:
                    rp = np.random.choice(N, 2)
                    A, B = X[rp[0], :], X[rp[1], :]
                    self.H[f'H_{i}'] = np.hstack((A - B, ( np.dot(A, B.T) - np.dot(A, A.T))/2))                                        
                    left = np.dot(np.hstack((X, np.ones((N, 1)))),self.H[f'H_{i}'].T) > 0
                    classificador_left = self.base_estimator
                    classificador_right = self.base_estimator
                    self.E[f'D_{i}']['left'] = classificador_left.fit(X[left, :], y[left])
                    self.E[f'D_{i}']['right'] = classificador_right.fit(X[~left, :], y[~left])  

                    #self.estimators_.append(classificador_left)                  
                    #self.estimators_.append(classificador_right)                  
                    break
                except:
                    continue
                break
                

        return self.E, self.H

    def _predict(self, X, y):

        L = len(self.E)
        ens = np.zeros((X.shape[0], L))

        for i in range(L):
            left = np.dot(np.hstack((X, np.ones((X.shape[0], 1)))), self.H[f'H_{i}'].T) > 0
            ens[left, i] = self.E[f'D_{i}']['left'].predict(X[left, :])
            ens[~left, i] = self.E[f'D_{i}']['right'].predict(X[~left, :])

        a1 = stats.mode(ens.T)[0]
        e = np.mean(a1 != y)

        return (ens, a1, e)
    
    def predict(self, X, y):
        predictions, _, _ = self._predict(X, y)
        pred_oracle = np.any((predictions == y.reshape(-1,1)), axis=1)
        return pred_oracle
    
    def score(self, X, y):
        pred_oracle = self.predict(X, y)
        acc_oracle = pred_oracle.sum() / pred_oracle.shape[0]
        return acc_oracle