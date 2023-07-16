from algo import *
from optimization import Bayesian_optimization
import numpy as np
from model import Model
from multiprocessing import Process, Manager
import concurrent.futures

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
    
    def evaluate_algo(self, algo, data_path, task, mode):
        print(f'Evalauting {algo.__name__} Model')
        bo = Bayesian_optimization(data_path, task, algo, mode=mode, default_params=True)
        best, X, y = bo.optimize(steps=3, out_log=False, return_trace=True)
        score = np.mean(y)
        print(y, score)
        return score

    def select(self, data_path, task, mode):
        score_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_algo = {executor.submit(self.evaluate_algo, algo, data_path, task, mode): algo for algo in self.algos}
            
            for future in concurrent.futures.as_completed(future_to_algo):
                algo = future_to_algo[future]
                try:
                    score = future.result()
                    score_list.append(score)
                except Exception as exc:
                    print(f'Error while evaluating {algo.__name__} Model: {exc}')
        
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
    