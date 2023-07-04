from optimization import Bayesian_optimization
from algo import rfcAlgo, mlpAlgo, logiAlgo, svmAlgo
from model import Model
from method import Method_select
import os


if __name__=="__main__":
    data_dir = 'Bayesian_main/data_benchmark/common'
    path_list = os.listdir(data_dir)
    for name in path_list:
        
        path = data_dir + '/' + name
        print(path)

        #Automatically decide an algorithm
        # algo_list = [rfcAlgo, svmAlgo, logiAlgo]
        # method = Method_select(algo_list)
        # score = method.select(data_path=path, task=Model)
        # algo = algo_list[score.index(max(score))]
        # print("Use" + str(algo)) 

        # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=rfcAlgo, 
            mode='radom',
            metrics=['auc', 'f1', 'pre'],
            default_params= True
            )
        
        x_best = bo.optimize(20)
        