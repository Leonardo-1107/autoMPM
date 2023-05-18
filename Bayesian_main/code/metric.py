import pickle
from itertools import combinations

import numpy as np
from algo import rfcAlgo, mlpAlgo
from model import Model

"""
OUT-OF-DATE-WARNING: The intersurface in this file may be out of date, requiring checking and updating!
"""

def get_shapley(dataset=None, data_path=None, params={}, algorithm=rfcAlgo, epoches=5, out_log=True):
    """Get shapley value of each feature in the dataset

    Args:
        dataset (tuple, optional): Features, labels, mask and feature name list. Prior to data_path. Defaults to None.
        data_path (str, optional): Path of input data file. Only used when dataset is None. Defaults to None.
        params (dict, optional): The param configuration used. Defaults to {}.
        algorithm (class, optional): The algorithm for the target task. Defaults to rfcAlgo.
        epoches (int, optional): The number of repeated trials for evaluation. Defaults to 5.
        out_log (bool, optional): Print the log if True. Defaults to True.

    Raises:
        RuntimeError: Both dataset and data_path are None.

    Returns:
        list: Shapley value of each feature in the dataset.
    """
    if dataset is not None:
        feature_arr, label_arr, common_mask, feature_name_list = dataset
    elif data_path is not None:
        with open(data_path, 'rb') as f:
            feature_arr, label_arr, common_mask, feature_name_list = pickle.load(f)
    else:
        raise RuntimeError("No data is specified in get_shapley!")
    
    feature_len = feature_arr.shape[1]
    score_dict = {}
    for i in range(feature_len+1):
        comb = [c for c in combinations(range(feature_len),i)]
        
        for c in comb:
            if c == ():
                score_dict[c] = 0
                continue
            
            score_list = []
            model = Model(feature_arr[:, c], label_arr, common_mask, algorithm)
            for _ in range(epoches):
                score = model.train(params=params)
                score_list.append(score)
            score_dict[c] = np.mean(score_list)
    
    shapley_list = []
    omit = [c for c in combinations(range(feature_len), feature_len-1)]
    for i in range(feature_len):
        shapley = 0
        o = omit[feature_len-i-1]
        for j in range(feature_len):
            omit_comb = [c for c in combinations(o,j)]
            for oc in omit_comb:
                oc_set = set(oc)
                oc_set.add(i)
                tup = tuple(oc_set)
                shapley += (score_dict[tup] - score_dict[oc]) / len(omit_comb)
        shapley = shapley / feature_len
        shapley_list.append(shapley)
    
    # output the shapley value of each features
    if out_log:
        for i in range(len(feature_name_list)):
            print(f'{feature_name_list[i]}:   \t{shapley_list[i]:+.4f}')
            
    return shapley_list