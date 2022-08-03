import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Return a dataframe with the selected columns
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.columns]
    
class IsAlone(BaseEstimator, TransformerMixin):
    """
    By including the feature 'SibSp' and 'Parch', Create a new feature indicating 
    if a passenger travels alone or not.
    """
    def __init__(self, feature_one, feature_two, feature_new):
        self.feature_one = feature_one
        self.feature_two = feature_two
        self.feature_new = feature_new
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df.loc[:, self.feature_new] = df.loc[:, self.feature_one] + df.loc[:, self.feature_two] + 1
        df.loc[:, self.feature_new] = df.loc[:, self.feature_new].map(lambda x: 1 if x == 1 else 0)
        return df
    
class ExtractName(BaseEstimator, TransformerMixin):
    """
    Extract the titles from the feature 'Name'
    """
    
    def __init__(self, feature_name):
        self.feature_name = feature_name
            
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df.loc[:, self.feature_name] = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]
        other_titles = ['Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Dona',
                        'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer']
        df.loc[:, self.feature_name] = df.loc[:, self.feature_name].replace(other_titles, 'Other')
        df.loc[:, self.feature_name] = df.loc[:, self.feature_name].map({"Master":0, "Miss":1, "Mrs" : 1 , "Mr":0, "Other":2})
        
        return df