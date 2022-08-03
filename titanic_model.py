import numpy as np
import pandas as pd
import dill

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from tempfile import mkdtemp
from shutil import rmtree

from custom_estimators import FeatureSelector, IsAlone, ExtractName


def train_model(serialize = True, file_name = 'titanic_model.dill'):
    '''
    Train a machine learning model to predict survival on the Titanic.

    Parameters:
        serialize: Whether to persist model to the memory.
        file_name: File name to use when persisting the model.

    Returns:
        best_model : The best model after hyperparameter tuning.
    '''

    # load data
    train_data = pd.read_csv('titanic/train.csv')
    train_data['Embarked'].fillna(value = train_data['Embarked'].value_counts().index[0], inplace=True)
    X = train_data.drop('Survived', axis = 1)
    y = train_data['Survived']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    clf = SVC(gamma = 'scale', probability=True)
    
    # columns selected
    num_columns = ['Age', 'Fare']
    cat_columns = ['Pclass', 'Sex', 'Embarked', 'IsAlone']

    # Pipeline
    num_pipe = Pipeline([('num_selector', FeatureSelector(num_columns)),
                     ('imputer', SimpleImputer(strategy = 'median')),
                     ('Normalization', StandardScaler())])
    cat_pipe = Pipeline([('cat_selector', FeatureSelector(cat_columns)),
                     ('ohe', OneHotEncoder(sparse = False))])
    union = FeatureUnion([('num', num_pipe),
                      ('cat', cat_pipe)])
    
    cache = mkdtemp()
    pipe = Pipeline([('IsAlone', IsAlone("SibSp", "Parch", 'IsAlone')),
                     ('union', union),
                     ('classifier', clf)])

    # Hyperparameter tuning
    #param_grid = [{'classifier__kernel': ['rbf'], 'classifier__gamma': [.1, .5, 1, 2, 5, 10],
    #               'classifier__C': [.1, 1, 10, 100, 1000]},
    #              {'classifier__kernel': ['linear'], 'classifier__C': [.1, 1, 10, 100, 1000]},
    #              {'classifier__kernel': ['poly'], 'classifier__degree': [2, 3, 4, 5],
    #               'classifier__C': [.1, 1, 10, 100, 1000]}]

    #grid_search = GridSearchCV(pipe, param_grid, cv = 5, scoring = 'accuracy', n_jobs = None, verbose = 1)
    pipe.fit(X_train, y_train)
    best_model = pipe
    rmtree(cache)

    # Persist the model
    if serialize:
        with open(file_name, 'wb') as f:
            dill.dump(best_model, f)

    return best_model


def deploy_model(df, file_name = 'titanic_model.dill'):
    '''
    Make predictions using the trained model.

    Parameters:
        df: dataframe.
        file_name : File name to use when persisting the model.

    Returns:
        proba : array, shape = (len(df), 2)
        Returns probability of not surviving and surviving, respectively.
    '''

    # Create the model if not persisted
    try:
        with open(file_name, 'rb') as f:
            model = dill.load(f)
    except FileNotFoundError:
        print("Model not found, creating...")
        train_model(serialize = True, file_name = file_name)
        return deploy_model(df, file_name = file_name)

    return model.predict_proba(df)

if __name__ == '__main__':
    train_model()
