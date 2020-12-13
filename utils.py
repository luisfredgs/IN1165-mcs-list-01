from sklearn.datasets import load_digits, load_wine
from typing import List, Tuple, Any
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def data_digits() -> Tuple[List, List]:
    X, y = load_digits(return_X_y=True)
    return 'digits', X, y

def data_wine() -> Tuple[List, List]:
    X, y = load_wine(return_X_y=True)
    return 'wine', X, y

def data_spam() -> Tuple[List, List]:
    """Dataset found on Kaggle website:
    https://www.kaggle.com/somesh24/spambase"""
    
    # normalizer = Normalizer(copy=False)
    sc = StandardScaler()

    df = pd.read_csv("./datasets/spambase.csv")
    y = df['class'].values
    X = df.drop(['class'], axis=1).to_numpy()
    X = sc.fit_transform(X)
    return 'spam', X,y

def data_creditcardfraud() -> Tuple[List, List]:
    """Dataset found on Kaggle website:
    https://www.kaggle.com/mlg-ulb/creditcardfraud"""

    # normalizer = Normalizer(copy=False)
    sc = StandardScaler()

    df = pd.read_csv("./datasets/creditcard.csv")
    X = df.drop(['Class'], axis=1).to_numpy()
    X = sc.fit_transform(X)
    y = df['Class'].values
    return 'credit_card_fraud', X, y


def data_customer() -> Tuple[List, List]:
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