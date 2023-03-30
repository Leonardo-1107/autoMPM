import pickle
import warnings
from inspect import isfunction
from multiprocessing import Process, Queue
from sklearn.preprocessing import StandardScaler

import numpy as np
from algo import rfcAlgo, logiAlgo, svmAlgo, mlpAlgo
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, plot_roc_curve
from sklearn.model_selection import train_test_split

from utils import show_result_map, getDepositMask

import sys
sys.setrecursionlimit(3000)



class Model:
    DEFAULT_METRIC = roc_auc_score
    DEFAULT_FIDELITY = 5
    DEFAULT_TEST_CLUSTERS = 4
    WARNING_FIDELITY_HIGH = 20
    
    def __init__(self, data_path, fidelity=3, test_clusters=4, algorithm=rfcAlgo, mode='random'):
        """
            This file was the model of original maching learning method
            Used for debugging the model of auto-ml

            Nova Scotia : Dataset for testing model performance

            Most of the file is the same to model.py and the explanation of functions involved can 
            be reached in that file
        """
        with open(data_path, 'rb') as f:
            feature_arr, total_label, common_mask, feature_name_list = pickle.load(f)
        self.set_fidelity(fidelity)
        self.set_test_clusters(test_clusters)
        self.feature_arr = feature_arr
        self.total_label_arr = total_label.astype(int)
        self.label_arr = self.total_label_arr[0]
        self.common_mask = common_mask
        self.height, self.width = common_mask.shape
        self.algorithm = algorithm
        self.path = data_path
        self.mode = mode
        return

    def set_test_clusters(self, test_clusters=DEFAULT_TEST_CLUSTERS):
        if not isinstance(test_clusters, int):
            raise RuntimeError("The test_clusters must be an integer!")
        if test_clusters <= 1:
            raise RuntimeError(f"The test_clusters must be more than 1, but now it is {test_clusters}!")
            
        self.test_clusters = test_clusters
        
    def set_fidelity(self, fidelity=DEFAULT_FIDELITY):
        if not isinstance(fidelity, int):
            raise RuntimeError("The fidelity must be an integer!")
        if fidelity < 1:
            raise RuntimeError(f"The fidelity must be positive, but now it is {fidelity}!")
        if fidelity > Model.WARNING_FIDELITY_HIGH:
            warnings.warn(f"The fidelity is suspiciously high. It is {fidelity}.")
            
        self.fidelity = fidelity
        
    def km(self, x, y, cluster):

        coord = np.concatenate([x, y], axis=1)
        cl = KMeans(n_clusters=cluster, random_state=0).fit(coord)
        cll = cl.labels_
        
        return cll

    def test_extend(self, x, y, test_num):

        # Build the test mask
        test_mask = np.zeros_like(self.common_mask).astype(bool)
        test_mask[x, y] = True

        candidate = set([])
        for i in range(test_num-1):
            # Add the neighbor grid which is in the valid region and not chosen yet into the candidate set
            if x >= 1 and self.common_mask[x-1, y] and not test_mask[x-1, y]:
                candidate.add((x-1, y))
            if y >= 1 and self.common_mask[x, y-1] and not test_mask[x, y-1]:
                candidate.add((x, y-1))
            if x <= self.height-2 and self.common_mask[x+1, y] and not test_mask[x+1, y]:
                candidate.add((x+1, y))
            if y <= self.width-2 and self.common_mask[x, y+1] and not test_mask[x, y+1]:
                candidate.add((x, y+1))
            
            # Randomly choose the next grid to put in the test set
            pick = np.random.randint(0, len(candidate))
            x, y = list(candidate)[pick]
            candidate.remove((x,y))
            test_mask[x, y] = True
            
        return test_mask
            
    def dataset_split(self, test_mask_list=None, modify=False, kmeans = True):

        if kmeans:
            if test_mask_list is None:
                test_mask_list = []
                # Randomly choose the start grid
                mask_sum = self.common_mask.sum()
                test_num = int(mask_sum / self.test_clusters)
                x_arr, y_arr = self.common_mask.nonzero()
                positive_x = x_arr[self.label_arr.astype(bool)].reshape((-1,1))
                positive_y = y_arr[self.label_arr.astype(bool)].reshape((-1,1))
                cll = self.km(positive_x, positive_y, self.test_clusters)
                
                for i in range(self.test_clusters):
                    cluster_arr = (cll == i)
                    cluster_x = positive_x[cluster_arr].squeeze()
                    cluster_y = positive_y[cluster_arr].squeeze()
                    # Throw out the empty array
                    if len(cluster_x.shape) == 0:
                        continue
                    
                    start = np.random.randint(0, cluster_arr.sum())
                    x, y = cluster_x[start], cluster_y[start]
                    test_mask = self.test_extend(x, y, test_num)
                    test_mask_list.append(test_mask)
            else:
                for test_mask in test_mask_list:
                    assert test_mask.shape == self.common_mask.shape

        # Buf the test mask
        tmpt = test_mask_list

        # Split the dataset
        dataset_list = []
        for test_mask in test_mask_list:
            train_mask = ~test_mask
            test_mask = test_mask & self.common_mask
            test_mask = test_mask[self.common_mask]
            train_mask = train_mask & self.common_mask
            train_mask = train_mask[self.common_mask]
            X_train_fold, X_test_fold = self.feature_arr[train_mask], self.feature_arr[test_mask]
            y_train_fold, y_test_fold = self.label_arr[train_mask], self.label_arr[test_mask]
            
            # Modify y_test_fold
            if modify:
                true_num = y_test_fold.sum()
                index = np.arange(len(y_test_fold))
                true_test = index[y_test_fold == 1]
                false_test = np.random.permutation(index[y_test_fold == 0])[:true_num]
                test = np.concatenate([true_test, false_test])
                X_test_fold = X_test_fold[test]
                y_test_fold = y_test_fold[test]
                
            dataset = (X_train_fold, y_train_fold, X_test_fold, y_test_fold)
            dataset_list.append(dataset)
        
        return tmpt, dataset_list
    
    def random_spilt(self, rate =0.2):

        feature = self.feature_arr
        total_label = self.total_label_arr
        ground_label = total_label[0]
        aug_label = total_label[1]
        index = np.arange(len(aug_label))
        
        train_index, test_index = train_test_split(
            index, 
            test_size=rate,
            shuffle=True)

        X_train_fold, X_test_fold, y_train_fold, y_test_fold = [],[],[],[]
        for i in train_index:
            X_train_fold.append(feature[i])
            y_train_fold.append(aug_label[i])
        
        for i in test_index:
            X_test_fold.append(feature[i])
            y_test_fold.append(ground_label[i])
        
        return X_train_fold, X_test_fold, y_train_fold, y_test_fold
        
            
    def train(self, params,  metrics=['auc','f1', 'pre'], test_mask=None, modify=False):

        if not isinstance(metrics, list):
            metrics = [metrics]
        metric_list = []

        for metric in metrics:
            if isinstance(metric, str):
                if metric.lower() == 'roc_auc_score' or metric.lower() == 'auc' or metric.lower() == 'auroc':
                    metric = roc_auc_score
                elif metric.lower() == 'roc_curve' or metric.lower() == 'roc':
                    metric = roc_curve
                elif metric.lower() == 'f1_score' or metric.lower() == 'f1':
                    metric = f1_score
                elif metric.lower() == 'precision_score' or metric.lower() == 'pre':
                    metric = precision_score
                elif metric.lower() == 'plot_roc' or metric.lower() == 'plot':
                    metric = plot_roc_curve    
                else:
                    warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                    metric = Model.DEFAULT_METRIC
            elif isfunction(metric):
                metric = metric
            else:
                warnings.warn(f'Wrong metric! Replace it with default metric {Model.DEFAULT_METRIC.__name__}.')
                metric = Model.DEFAULT_METRIC
            metric_list.append(metric)
            

        score_list = []
        if self.mode  == 'random':
            # N fold-cross
            for i in range(self.test_clusters):
                X_train_fold, X_test_fold, y_train_fold, y_test_fold = self.random_spilt(rate=1/self.test_clusters) 
                algo = self.algorithm(params)
                algo.fit(X_train_fold, y_train_fold)
                pred_arr, y_arr = algo.predicter(X_test_fold)
                scores = []
                for metric in metric_list:
                    if metric == f1_score or metric == precision_score:
                        # Only make sense when data augment deployed
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score) 
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)
                    
                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)

        else: 
            test_mask, dataset_list = self.dataset_split(test_mask, modify)
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_test_fold, y_test_fold = dataset
                algo = self.algorithm(params)
                algo.fit(X_train_fold, y_train_fold)
                pred_arr, y_arr = algo.predicter(X_test_fold)
                
                scores = []
                for metric in metric_list:
                    if metric == f1_score or metric == precision_score:
                        # Only make sense when data augment deployed
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score) 
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)
                    
                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)

        return score_list
    
    def obj_train_parallel(self, queue, args):

        score = self.train(args)
        queue.put(score)
        return
    
    def evaluate(self, x):

        queue = Queue()
        process_list = []
        for _ in range(self.fidelity):
            p = Process(target=self.obj_train_parallel, args=[queue, x])
            p.start()
            process_list.append(p)
            
        for p in process_list:
            p.join()
            
        score_list = []
        for i in range(self.fidelity):
            score_list.append(queue.get())

        y = np.mean(
            np.mean(score_list, axis=0),
            axis=0
            )
        std_arr = np.std(score_list, axis=1)
        auc_std, f_std = np.mean(std_arr[:,0]), np.mean(std_arr[:,1])
        y = list(y) + [auc_std, f_std]
        return y

# For debugging
if __name__ == '__main__':

    alg = rfcAlgo
    x = {}
    if alg == svmAlgo:
        x = {'probability': True}

    task = Model(
        data_path='Bayesian_main/data/tok_lad_scsr_ahc.pkl',
        algorithm=alg,
        mode='random'
        )
    
    y = task.evaluate(x)
    print(y)
    