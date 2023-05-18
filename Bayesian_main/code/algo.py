from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
"""
The encapsulation of algorithms.

Require a parameter in __init__ as the params of the actual model.
Require a function named predict, taking input data and outputing confidence on the positive label
Recommend default param settings as static members of class definition
"""

class rfcAlgo(RandomForestClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [20, 150], 'max_depth': [10, 50]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {} # 'random_state': 0
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X, save_weight = False):
        # Output the pred layer and the proba layer simultaneously
        if save_weight:
            np.save('Bayesian_main/weights.npy', self.feature_importances_)
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
    
class logiAlgo(LogisticRegression):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {}
    DEFAULT_ENUM_BOOK = {
        'penalty':['l1','l2'], 
        'solver': ["liblinear", "saga"]
        }
    DEFAULT_STATIC_BOOK = {'max_iter':1000}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params
    
    def predicter(self, X):   
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
        
class mlpAlgo(MLPClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate_init': [0.0001, 0.01]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(100,), (100,256)]}  # 'solver': ['lbfgs', 'sgd' ,'adam']
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]

class svmAlgo(SVC):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'kernel':['poly', 'rbf']} 
    DEFAULT_STATIC_BOOK = {'probability': True}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        
    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]
