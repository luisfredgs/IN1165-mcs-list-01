from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from typing import List, Tuple
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class Dataset_Loader():

    @staticmethod
    def car() -> Tuple[str, List, List]:
        """The database evaluates cars according to the following concepts: car acceptability, 
        overall price, buying price, price of the maintenance, number of doors, capacity in 
        terms of persons to carry, the size of luggage boot, and estimated safety of the car.
        """
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data=pd.read_csv(url, header=None, )
        X = data.iloc[:,:-1]
        X = pd.get_dummies(X)
        
        sc = StandardScaler()
        X = sc.fit_transform(X.to_numpy())

        y = data.iloc[:,-1]
        y = y.astype('category').cat.codes
        return 'car', X, y.values   
                                      
    
    @staticmethod
    def seismic() -> Tuple[str, List, List]:
        """Mining activity was and is always connected with the occurrence of dangers which 
        are commonly called mining hazards. In this case, the goal is to predict hazardous 
        seismic activity
        """
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
        data=pd.read_csv(url, skiprows=154, header=0, sep=',')
        X = data.iloc[:,:-1]
        X = pd.get_dummies(X)
        sc = StandardScaler()
        X = sc.fit_transform(X.to_numpy())
        y = data.iloc[:,-1]
        y = y.astype('category').cat.codes
        return 'seismic', X, y.values                              
      
    
    @staticmethod
    def data_customer() -> Tuple[str, List, List]:
        """Dataset found on Kaggle website:
        kaggle.com/prathamtripathi/customersegmentation"""

        # normalizer = Normalizer(copy=False)
        sc = StandardScaler()

        df = pd.read_csv("./datasets/customer_segmentation.csv")
        X = df.drop(['custcat'], axis=1).to_numpy()
        X = sc.fit_transform(X)

        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(df['custcat'].values)
        return 'customer', X, y 
    
    @staticmethod
    def diabetic_retinopathy() -> Tuple[str, List, List]:
        """This dataset contains features extracted from the Messidor image set to predict 
        whether an image contains signs of diabetic retinopathy or not. All features represent 
        either a detected lesion, a descriptive feature of a anatomical part or an image-level descriptor.
        """
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
        data=pd.read_csv(url, skiprows=24, header=None)
        X = data.iloc[:,:-1]
        X = pd.get_dummies(X)
        sc = StandardScaler()
        X = sc.fit_transform(X.to_numpy())
        y = data.iloc[:,-1]
        y = y.astype('category').cat.codes
        return 'diabetic_retinopathy', X, y.values
    

    # @staticmethod
    # def data_wine() -> Tuple[List, List]:
    #     X, y = load_wine(return_X_y=True)
    #     return 'wine', X, y

    @staticmethod
    def seeds() -> Tuple[str, List, List]:
        """
        The examined group comprised kernels belonging to three different 
        varieties of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected 
        for the experiment. High quality visualization of the internal kernel structure was 
        detected using a soft X-ray technique. It is non-destructive and considerably cheaper 
        than other more sophisticated imaging techniques like scanning microscopy or laser technology. 
        The images were recorded on 13x18 cm X-ray KODAK plates. Studies were conducted using combine 
        harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics 
        of the Polish Academy of Sciences in Lublin.
        """
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        data=pd.read_csv(url, header=0, sep='\s+')
        X = data.iloc[:,:-1]
        X = pd.get_dummies(X)
        sc = StandardScaler()
        X = sc.fit_transform(X.to_numpy())
        y = data.iloc[:,-1]
        y = y.astype('category').cat.codes
        return 'seeds', X, y.values