from algo import rfcAlgo, logiAlgo, mlpAlgo, svmAlgo
from optimization import Bayesian_optimization
import numpy as np
from model import Model

"""
Here is a new method to choose a better machine learning model before the main preocess to optimize
To balance the performance and time consumption, Each method to be selected will be iterated 3 times, 
and the best score obtained will be used as the evaluation criterion .The model that gets the highest
score will be selected in the follow up process

This feature can be enabled in test.py by calling Method_select class
"""

class Method_select:
    def __init__(self, algorithms=[rfcAlgo, mlpAlgo]):
        self.algos = algorithms
    
    def select(self, data_path, task):
        score_list = []
        for algo in self.algos:
            print(f'Evalauting {str(algo)} Model')
            bo = Bayesian_optimization(data_path, task, algo,'random', True)
            best, X, y = bo.optimize(steps=3, out_log=False, return_trace=True)
            score = np.max(y)
            score_list.append(score)

        return score_list

# For test
if __name__=="__main__":
    method = Method_select()
    algo_list = [rfcAlgo, mlpAlgo, logiAlgo]
    score = method.select(data_path='./data/nefb_fb_hlc_cir.pkl', task=Model)
    print(score)
    
    bo = Bayesian_optimization('./data/nefb_fb_hlc_cir.pkl', algorithm=algo_list[score.index(max(score))], default_params= True)
    x_best = bo.optimize(20)
    print(f'Best   n_estimators: {x_best[0]}, max_depth: {x_best[1]}, criterion: {x_best[2]}')
    