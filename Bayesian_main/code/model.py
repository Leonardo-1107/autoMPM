import pickle
import warnings
from inspect import isfunction
from multiprocessing import Process, Queue

import numpy as np
from algo import rfcAlgo, mlpAlgo
from sklearn.cluster import KMeans
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, accuracy_score, recall_score
from sklearn.model_selection import KFold, train_test_split

from utils import *

class Model:
    DEFAULT_METRIC = roc_auc_score
    DEFAULT_FIDELITY = 5
    DEFAULT_TEST_CLUSTERS = 4
    WARNING_FIDELITY_HIGH = 20
    
    def __init__(self, data_path, fidelity=3, test_clusters=4, algorithm=rfcAlgo, mode='random', metrics =['f1','pre','auc'], modify = False):
        """Initialize the input data, algorithm for the target task, evaluation fidelity and test clusters

        Args:
            data_path (str): The path of input data files
            fidelity (int, optional): The number repeated trials for evaluation. Defaults to 3.
            test_clusters (int, optional): The number of clusters in data split. Defaults to 4.
            algorithm (_type_, optional): The algorithm used to accomplish the target task. Defaults to rfcAlgo.
            metrics: (list, optional): The metrics used to judege the performance.
        """
        with open(data_path, 'rb') as f:
            feature_arr, total_label, common_mask, deposit_mask = pickle.load(f)
        self.set_fidelity(fidelity)
        self.set_test_clusters(test_clusters)
        self.feature_arr = feature_arr
        self.total_label_arr = total_label.astype(int)
        self.label_arr = self.total_label_arr[0]
        self.common_mask = common_mask
        self.deposit_mask = deposit_mask
        self.height, self.width = common_mask.shape
        self.algorithm = algorithm
        self.path = data_path
        self.mode = mode
        self.test_index = 0
        self.metrics = metrics
        self.modify = modify
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
        """Clustering the positive samples with k-means

        Returns:
            array: The cluster id that each sample belongs to
        """
        coord = np.concatenate([x, y], axis=1)
        cl = KMeans(n_init='auto', n_clusters=cluster).fit(coord)
        cll = cl.labels_
        
        return cll

    def test_extend(self, x, y, test_num):
        """Extend from the start point to generate the tesk mask

        Args:
            x (int): The x coord of the start point to extend from.
            y (int): The y coord of the start point to extend from.
            test_num (_type_): The size of test set

        Returns:
            Array: Mask for test set
        """
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
            
    def dataset_split(self, test_mask_list=None, modify=False):
        """Split the dataset into train set and test set, repeated for the number of clusters

        Args:
            test_mask_list (list, optional): Pre-prepared test masks if provided. Defaults to None.
            modify (bool, optional): Undersample if True. Defaults to False.

        Returns:
            list: test mask list
            list: splited dataset list
        """
        
        if test_mask_list is None:
            test_mask_list = []
            # Randomly choose the start grid
            mask_sum = self.common_mask.sum()
            test_num = int(mask_sum / 5)
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
            y_train_fold, y_test_fold = self.total_label_arr[1][train_mask], self.label_arr[test_mask]

            # split the val set
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                                                                    X_train_fold, 
                                                                    y_train_fold, 
                                                                    test_size=0.25, 
                                                                    random_state=42)
            # Modify y_test_fold
            if modify:
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]
            
            X_train_fold, y_train_fold = sk_shuffle(X_train_fold, y_train_fold)

            dataset = (X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold)
            dataset_list.append(dataset)
        
        return tmpt, dataset_list
    
    def random_split(self, modify =False):
        """Split the dataset into train set and test set, and apply K-fold Cross-validation.

        Args:
            modify (bool, optional): Undersample if True. Defaults to False.

        Returns:
            list: splited dataset list
        """
        
        feature = self.feature_arr
        total_label = self.total_label_arr
        ground_label = total_label[0]
        aug_label = total_label[1]
        
        dataset_list = []
        kf = KFold(n_splits=5, shuffle=True)
        for train_index , test_index in kf.split(feature): 
            
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = [],[],[],[]
            X_val_fold, y_val_fold = [], []
            
            for i in train_index:
                X_train_fold.append(feature[i])
                y_train_fold.append(aug_label[i])

            # split the val set
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                                                                    X_train_fold, 
                                                                    y_train_fold, 
                                                                    test_size=0.25, 
                                                                    random_state=42)

            for i in test_index:
                X_test_fold.append(feature[i])
                y_test_fold.append(ground_label[i])

            X_train_fold, X_test_fold, y_train_fold, y_test_fold = np.array(X_train_fold), np.array(X_test_fold), np.array(y_train_fold), np.array(y_test_fold)
            X_val_fold, y_val_fold = np.array(X_val_fold), np.array(y_val_fold)

            if y_test_fold.sum() == 0: 
                continue
            if modify:
                    true_num = y_train_fold.sum()
                    index = np.arange(len(y_train_fold))
                    true_train = index[y_train_fold == 1]
                    false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                    train = np.concatenate([true_train, false_train])
                    X_train_fold = X_train_fold[train]
                    y_train_fold = y_train_fold[train]

            dataset = (X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold)
            dataset_list.append(dataset)
            
        return dataset_list 

    def train(self, params, test_mask=None, modify=False):
        """Train a model with test-set as a rectangle

        Args:
            params (dict): parameters of the machine learning algorithm
            metrics (list, optional): List of metrics used for evaluation. Defaults to roc_auc_score.
            test_mask (Array, optional): The pre-prepared test mask if provided. Defaults to None.
            modify (bool, optional): Undersample if True. Defaults to False.

        Returns:
            score_list (list): Scores for each metric
        """
        modify = self.modify
        metrics = self.metrics
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
                elif metric.lower() == 'recall_score' or metric.lower() == 'recall':
                    metric = recall_score
     
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
            dataset_list = self.random_split(modify=False)
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset
                algo = self.algorithm(params)
                algo.fit(X_train_fold, y_train_fold)
                scores = []
                
                loss = criterion_loss(self.mode, algo, X_val_fold, y_val_fold, y_train_fold)
                scores.append(loss)
                    
                # testing
                pred_arr, y_arr = algo.predicter(X_test_fold)
             
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score) 
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)

                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)
            
        else: 
            test_mask_list, dataset_list = self.dataset_split(test_mask, modify=True)
            y_arr_record = []
            for dataset in dataset_list:
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_fold, y_test_fold = dataset
                algo = self.algorithm(params)
                algo.fit(X_train_fold, y_train_fold)
                scores = []
                
                loss = criterion_loss(self.mode, algo, X_val_fold, y_val_fold, y_train_fold)
                scores.append(loss)
                
                # testing
                pred_arr, y_arr = algo.predicter(X_test_fold)
                
                for metric in metric_list:
                    if metric != roc_auc_score:
                        score = metric(y_true=y_test_fold, y_pred=pred_arr)
                        scores.append(score)
                    else:
                        score = metric(y_test_fold, y_arr)
                        scores.append(score)    

                if len(scores) == 1:
                    scores = scores[0]
                score_list.append(scores)

                _, y_arr_feature_map = algo.predicter(self.feature_arr)
                y_arr_record.append(y_arr_feature_map)
            
            # plot result map and confusion matrix
            # for i in range(len(test_mask_list)):   
                # show_result_map(result_values=y_arr_record[i], 
                #                 mask=self.common_mask, deposit_mask=self.deposit_mask, test_mask=test_mask_list[i], 
                #                 index=i+1, clusters=self.test_clusters)
        return score_list

    def obj_train_parallel(self, queue, args):
        """Encapsulation for parallelizing the repeated trials

        Args:
            queue (Queue): Store the output of processes
            args (_type_): Args for algorithm
        """
        score = self.train(args)
        queue.put(score)
        return
    
    def evaluate(self, x):
        """Evaluate the input configuration

        Args:
            x (iterable): Input configuration

        Returns:
            float: Score of the input configuration
        """
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
        try:
            score1_std, score2_std = np.mean(std_arr[:,0]), np.mean(std_arr[:,1])
            y = list(y) + [score1_std, score2_std]
        except:
            y = y      
        return y