from optimization import Bayesian_optimization
from algo import *
from model import Model
from method import Method_select
import os
import warnings
warnings.filterwarnings("ignore")
os.chdir('Bayesian_main')

if __name__=="__main__":

    ############### TODO ###################

    data_dir = 'data'                                                   # the location of dataset
    mode = 'OOD'                                                        # test mod. Default to be OOD

    optimization_steps = 100                                            # optimization steps. Default to be 100
    early_stop = 20                                                     # early stop number. Deafult to be 20

    ############### TODO END ###############

    path_list = os.listdir(data_dir)
    
    if mode == 'IID':
        metrics = ['f1', 'auc']
    elif mode == 'OOD':
        metrics = ['auc', 'f1', 'pre']
    else:
        print("WRONG MODE SETTING, END")
        exit(0)


    for name in path_list:
        path = data_dir + '/' + name

        # Automatically decide an algorithm
        algo_list = [rfcAlgo, extAlgo, svmAlgo, NNAlgo, gBoostAlgo]
        method = Method_select(algo_list)
        algo = method.select(data_path=path, task=Model, mode=mode)

        print(f"\n{name}, Use {algo.__name__}")
        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=algo, 
            mode=mode,
            metrics=metrics,                                      
            default_params= True
            )
        
        x_best = bo.optimize(optimization_steps, early_stop = early_stop)
        