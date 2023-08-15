from optimization import Bayesian_optimization
from algo import *
from model import Model
from method import Method_select
import os
import warnings
warnings.filterwarnings("ignore")
os.chdir('Bayesian_main')

if __name__=="__main__":
    data_dir = 'data/common'
    mode = 'random'
    path_list = os.listdir(data_dir)
    for name in path_list:
        path = data_dir + '/' + name
        print(name)

        # Automatically decide an algorithm
        # algo_list = [rfcAlgo, extAlgo, svmAlgo, NNAlgo, gBoostAlgo]
        # method = Method_select(algo_list)
        # algo = method.select(data_path=path, task=Model, mode=mode)
        algo = rfcAlgo
        print(f"\n{name}, Use {algo.__name__}")
        
        # algo = rfBoostAlgo
        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=algo, 
            mode=mode,
            metrics=['f1', 'auc'],
            default_params= True
            )
        
        x_best = bo.optimize(300, early_stop = 50)
        