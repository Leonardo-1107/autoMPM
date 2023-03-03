from optimization import Bayesian_optimization
from algo import rfcAlgo, mlpAlgo, logiAlgo, svmAlgo
from model import Model
from method import Method_select
import os


if __name__=="__main__":
    path = 'Bayesian_main/data/North_Idaho.pkl'

    # algo_list = [rfcAlgo, svmAlgo, logiAlgo]
    # method = Method_select(algo_list)
    # score = method.select(data_path=path, task=Model)
    # print("Use" + str()) algo_list[score.index(max(score))]

    bo = Bayesian_optimization(
        data_path=path, 
        algorithm=rfcAlgo, 
        default_params= True
        )
    
    x_best = bo.optimize(40)
    
    # for rfcAlgo estimation
    print(f'Best   n_estimators: {x_best[0]}, max_depth: {x_best[1]}, criterion: {x_best[2]}')
