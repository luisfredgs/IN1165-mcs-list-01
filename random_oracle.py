import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron

class RandomOracleModel(object):
    
    """It is an adaptation from original code
    https://github.com/ndinhtuan/oracle_ensemble"""

    def __init__(self):
        
        self.model1 = None
        self.model2 = None
        self.inst1 = None 
        self.inst2 = None

        self.model1 = Perceptron()
        self.model2 = Perceptron()
    
    def distance(self, x1, x2):

        a = np.array(x1)
        b = np.array(x2)
        return np.linalg.norm(a-b)


    def fit(self, X, Y):
        
        len_train = len(X)
        i1 = np.random.randint(len_train)
        i2 = np.random.randint(len_train)

        while i2==i1:
            i2 = np.random.randint(len_train)

        self.inst1 = inst1 = X[i1]
        self.inst2 = inst2 = X[i2]

        X1 = [] 
        Y1 = []
        X2 = [] 
        Y2 = []
        
        for x, y in zip(X, Y):
            if self.distance(x, inst1) < self.distance(x, inst2):
                X1.append(x)
                Y1.append(y)
            else:
                X2.append(x) 
                Y2.append(y)

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        X2 = np.array(X2)
        Y2 = np.array(Y2)
        
        
        print("Training Model 1")
        self.model1.fit(X1, Y1)
        print("Training Model 2")
        self.model2.fit(X2, Y2)
    
    def predict(self, x):

        assert self.model1 is not None and self.model2 is not None
        assert self.inst1 is not None and self.inst2 is not None
        
        if self.distance(x, self.inst1) < self.distance(x, self.inst2):
            return self.model1.predict([x])[0]
        else:
            return self.model2.predict([x])[0]

    def evaluate(self, x_test, y_test):

        assert self.model1 is not None and self.model2 is not None
        assert self.inst1 is not None and self.inst2 is not None

        preds = []

        for x in x_test:
            preds.append(self.predict(x)) 

        preds = np.array(preds)
        
        return classification_report(y_test, preds)