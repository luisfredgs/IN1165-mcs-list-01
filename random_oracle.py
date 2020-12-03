import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import Perceptron, SGDClassifier

class RandomOracleModel(object):
    
    """ In the Random Oracle ensemble method, each base classifier is a mini-ensemble of two classifiers 
    and a randomly generated oracle that selects one of the two classifiers.
    
    The oracle selects one of the two classifiers and can be seen as a random discriminant function.
    In this implementation, a linear oracle (Random Linear Oracles) divides the space into two subspaces 
    using a hyperplane. To build the oracle, two different training objects are selected at random, 
    the points that are at the same distance  from the two training objects define the hyperplane. 
    Each remaining training object is assigned to the subspace of the selected training object for 
    which is closer.
    
    This code is adapted from an implementation of "Random Oracle Ensembles for Imbalanced Data"
    avaliable in: https://github.com/ndinhtuan/oracle_ensemble

    References
    ----------
    Kuncheva L.I. and J.J. Rodriguez, Classifier ensembles with a random linear oracle, 
    IEEE Transactions on Knowledge and Data Engineering, 19 (4), 500-508, 2007.

    Rodríguez, Juan & Díez-Pastor, Jose-Francisco & García-Osorio, César. (2013). 
    Random Oracle Ensembles for Imbalanced Data. 7872. 247-258. 10.1007/978-3-642-38067-9_22. 
    
    """

    def __init__(self):
        
        self.classifier1 = None
        self.classifier2 = None
        self.instance_1 = None 
        self.instance_2 = None

        perceptron = SGDClassifier(loss="perceptron", eta0=1.e-17,max_iter=1, 
                                learning_rate="constant", penalty=None)
        self.classifier1 = perceptron
        self.classifier2 = perceptron
    

    def distance(self, x1, x2):
        """ The distances are calculated according to the Euclidean distance.
        The following call works because the Euclidean distance is equivalent 
        to the l2 norm.
        
        d(x,y) = \bigg( \sum^n_{k=1}|x_{k}-y_{k}|^r \bigg)^{1/r}
        
        """

        return np.linalg.norm(np.array(x1)-np.array(x2))


    def fit(self, X, Y):
        """Split the training data in two subsets. 
        For each subset of the training data, build a classifier.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.
        y : array of shape (n_samples)
            class labels of each example in X.

        """
        len_train = len(X)
        i1 = np.random.randint(len_train)
        i2 = np.random.randint(len_train)

        while i2==i1:
            i2 = np.random.randint(len_train)

        self.instance_1 = instance_1 = X[i1]
        self.instance_2 = instance_2 = X[i2]

        X1 = [] 
        Y1 = []
        X2 = [] 
        Y2 = []
        
        for x, y in zip(X, Y):
            if self.distance(x, instance_1) < self.distance(x, instance_2):
                X1.append(x)
                Y1.append(y)
            else:
                X2.append(x) 
                Y2.append(y)

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        X2 = np.array(X2)
        Y2 = np.array(Y2)
        

        self.classifier1.fit(X1, Y1)
        self.classifier2.fit(X2, Y2)
    
    def predict(self, x):
        """Use the Random Oracle to select one of the two classifiers and
        returns the prediction given by the selected classifier."""

        assert self.classifier1 is not None and self.classifier2 is not None
        assert self.instance_1 is not None and self.instance_2 is not None
        
        if self.distance(x, self.instance_1) < self.distance(x, self.instance_2):
            return self.classifier1.predict([x])[0]
        else:
            return self.classifier2.predict([x])[0]

    def score(self, x_test, y_test):

        assert self.classifier1 is not None and self.classifier2 is not None
        assert self.instance_1 is not None and self.instance_2 is not None

        preds = []

        for x in x_test:
            preds.append(self.predict(x)) 

        preds = np.array(preds)
        
        return accuracy_score(y_test, preds)