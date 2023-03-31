from optimization import Bayesian_optimization
from algo import rfcAlgo, mlpAlgo, logiAlgo, svmAlgo
from model import Model
from method import Method_select
import os


if __name__=="__main__":
    path_list = ['North_Idaho', 'bm_lis_go_sesrp', 'nefb_fb_hlc_cir', 'tok_lad_scsr_ahc', 'Washington']
    for name in path_list:
        path = f'Bayesian_main/data/{name}.pkl'

    # Automatically decide an algorithm
    # algo_list = [rfcAlgo, svmAlgo, logiAlgo]
    # method = Method_select(algo_list)
    # score = method.select(data_path=path, task=Model)
    # algo = algo_list[score.index(max(score))]
    # print("Use" + str(algo)) 

    # Bayesian optimization process
        bo = Bayesian_optimization(
            data_path=path, 
            algorithm=rfcAlgo, 
            mode='k',
            default_params= True
            )
        
        x_best = bo.optimize(20)
        
        # for rfcAlgo estimation
        print(f'Best   n_estimators: {x_best[0]}, max_depth: {x_best[1]}, criterion: {x_best[2]}')
