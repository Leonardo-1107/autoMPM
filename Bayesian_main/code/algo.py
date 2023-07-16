from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
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

class logiAlgo(LogisticRegression):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {}
    DEFAULT_ENUM_BOOK = {
        'penalty':['l1','l2'], 
        'solver': ["liblinear", "saga"]
        }
    DEFAULT_STATIC_BOOK = {'max_iter':2000}
    
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
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate_init': [0.0001, 0.1]}
    DEFAULT_DISCRETE_BOOK = {} 
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(50,), (100,), (256,), (100,256), (100, 256, 100)]}  # NAS
    DEFAULT_STATIC_BOOK = {'max_iter':2000, 'early_stopping': True}
    
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
    DEFAULT_ENUM_BOOK = {'tol': [1e-3, 1e-4, 1e-5], 'kernel':['poly', 'rbf']} 
    DEFAULT_STATIC_BOOK = {'max_iter':2000, 'probability': True}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params
        
    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:,1]

class rfcAlgo(RandomForestClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 100], 'max_depth': [5, 30]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {} 
    
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


class balanceRfcAlgo(BalancedRandomForestClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 100], 'max_depth': [5, 30]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {} 
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]
    
class extreeAlgo(ExtraTreesClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 100], 'max_depth': [5, 30]}
    DEFAULT_ENUM_BOOK = {'criterion': ['gini', 'entropy']}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

class gBoostAlgo(GradientBoostingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 80]}
    DEFAULT_ENUM_BOOK = {'loss' : ['log_loss', 'deviance', 'exponential']}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

class aBoostAlgo(AdaBoostClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'learning_rate': [0.01, 0.5]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 100]}
    DEFAULT_ENUM_BOOK = {}
    DEFAULT_STATIC_BOOK = {}
    
    def __init__(self, params):
        super().__init__(**params)
        self.params = params

    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]


class svmBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 50]}
    DEFAULT_ENUM_BOOK = {'tol': [1e-3, 1e-4, 1e-5], 'kernel':['poly', 'rbf']}
    DEFAULT_STATIC_BOOK = {}

    def __init__(self, params):
        svm_params = {k: v for k, v in params.items() if k in svmAlgo.DEFAULT_STATIC_BOOK}
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}

        base_estimator = svmAlgo(svm_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]
    

class logiBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {'C':[0.5,1.0]}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 50]}
    DEFAULT_ENUM_BOOK = {'solver': ["liblinear", "saga"]}
    DEFAULT_STATIC_BOOK = {'max_iter': 1000}

    def __init__(self, params):
        logi_params = {k: v for k, v in params.items() if k in logiAlgo.DEFAULT_STATIC_BOOK}
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}

        base_estimator = logiAlgo(logi_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]


class mlpBaggingAlgo(BaggingClassifier):
    DEFAULT_CONTINUOUS_BOOK = {}
    DEFAULT_DISCRETE_BOOK = {'n_estimators': [10, 20]}
    DEFAULT_ENUM_BOOK = {'hidden_layer_sizes':[(50,), (100,), (100,256)]}
    DEFAULT_STATIC_BOOK = {'max_iter': 1000}

    def __init__(self, params):
        mlp_params = {k: v for k, v in params.items() if k in mlpAlgo.DEFAULT_STATIC_BOOK}
        bagging_params = {k: v for k, v in params.items() if k in self.DEFAULT_DISCRETE_BOOK}

        base_estimator = mlpAlgo(mlp_params)
        super().__init__(estimator=base_estimator, **bagging_params)
        self.params = params
    
    def predicter(self, X):
        pred = self.predict(X)
        y = self.predict_proba(X)
        if isinstance(y, list):
            y = y[0]
        return pred, y[:, 1]

