from optimization import Bayesian_optimization
from algo import *
from model import Model
from method import Method_select
import os
import warnings
warnings.filterwarnings("ignore")

def autoMPM(data_dir, run_mode = 'IID', optimize_step = 40, metrics=['auc', 'f1', 'pre']):
    
    if run_mode == 'IID':
        mode = 'random'
    else:
        mode  = 'k_split'

    path_list = os.listdir(data_dir), 
    for name in path_list:
        path = data_dir + '/' + name

        # Automatically decide an algorithm
        algo_list = [rfcAlgo, svmAlgo, logiAlgo, mlpAlgo]
        method = Method_select(algo_list)
        score = method.select(data_path=path, task=Model, mode=mode)
        algo = algo_list[score.index(max(score))]
        print("Use" + str(algo)) 
        
        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=algo, 
            mode=mode,
            metrics=['auc', 'f1', 'pre'],
            default_params= True
            )
        
        x_best = bo.optimize(steps=optimize_step)

if __name__=="__main__":
    data_dir = 'Bayesian_main/data_benchmark/common'
    mode = 'randm'
    path_list = os.listdir(data_dir)
    for name in path_list:
        path = data_dir + '/' + name
        print(name)

        # Automatically decide an algorithm
        algo_list = [rfcAlgo, logiAlgo, svmAlgo, mlpAlgo, aBoostAlgo, gBoostAlgo, extreeAlgo, balanceRfcAlgo, logiBaggingAlgo, svmBaggingAlgo, mlpBaggingAlgo]
        algo_list = [aBoostAlgo, gBoostAlgo, extreeAlgo, balanceRfcAlgo, logiBaggingAlgo, svmBaggingAlgo, mlpBaggingAlgo]
        method = Method_select(algo_list)
        score = method.select(data_path=path, task=Model, mode=mode)
        algo = algo_list[score.index(max(score))]
        print("Use" + str(algo))
        break 
        algo = logiBaggingAlgo
        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=algo, 
            mode=mode,
            metrics=['auc', 'f1'],
            default_params= True
            )
        
        x_best = bo.optimize(40)
        