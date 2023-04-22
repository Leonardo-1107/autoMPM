import pickle
import warnings
from inspect import isfunction
from multiprocessing import Process, Queue

import numpy as np
from algo import rfcAlgo, mlpAlgo
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, plot_roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

from utils import show_result_map, getDepositMask
name = 'Bayesian_main/data/NovaScotia2.pkl'

class Model:
    DEFAULT_METRIC = roc_auc_score
    DEFAULT_FIDELITY = 5
    DEFAULT_TEST_CLUSTERS = 4
    WARNING_FIDELITY_HIGH = 20
    
    def __init__(self, data_path, fidelity=3, test_clusters=4, algorithm=rfcAlgo, mode='random'):
        """Initialize the input data, algorithm for the target task, evaluation fidelity and test clusters

        Args:
            data_path (str): The path of input data files
            fidelity (int, optional): The number repeated trials for evaluation. Defaults to 3.
            test_clusters (int, optional): The number of clusters in data split. Defaults to 4.
            algorithm (_type_, optional): The algorithm used to accomplish the target task. Defaults to rfcAlgo.
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
        self.test_index = 0
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
        cl = KMeans(n_clusters=cluster, random_state=0).fit(coord)
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
            test_num = int(mask_sum / self.test_clusters)
            x_arr, y_arr = self.common_mask.nonzero()
            
            positive_x = x_arr[self.label_arr.astype(bool)].reshape((-1,1))
            positive_y = y_arr[self.label_arr.astype(bool)].reshape((-1,1))
            
            cll = self.km(positive_x, positive_y, self.test_clusters)
            
            for i in range(self.test_clusters):
                cluster_arr = (cll == i)
                # print(i, cluster_arr.sum()/self.label_arr.sum())
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
            
            # Modify y_test_fold
            if modify:
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]
            
            dataset = (X_train_fold, y_train_fold, X_test_fold, y_test_fold)
            dataset_list.append(dataset)
        
        return tmpt, dataset_list
    
    def random_spilt(self, rate =0.2, modify =False):
        """Split the dataset into train set and test set, repeated for the number of clusters

        Args:
            rate: the ratio of the size of train and test dataset

        Returns:
            the spilt result with the Same form of modules in skit-learn
        """
        
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
            
        X_train_fold, X_test_fold, y_train_fold, y_test_fold = np.array(X_train_fold), np.array(X_test_fold), np.array(y_train_fold), np.array(y_test_fold)
        
        if modify:
                
                true_num = y_train_fold.sum()
                index = np.arange(len(y_train_fold))
                true_train = index[y_train_fold == 1]
                false_train = np.random.permutation(index[y_train_fold == 0])[:true_num]
                train = np.concatenate([true_train, false_train])
                X_train_fold = X_train_fold[train]
                y_train_fold = y_train_fold[train]
                
        return X_train_fold, X_test_fold, y_train_fold, y_test_fold
    
    
    def train(self, params,  metrics=['auc','f1','pre'], test_mask=None, modify=False):
        """Train a random forest with test-set as a rectangle

        Args:
            params (dict): parameters of the machine learning algorithm
            metrics (list, optional): List of metrics used for evaluation. Defaults to roc_auc_score.
            test_mask (Array, optional): The pre-prepared test mask if provided. Defaults to None.
            modify (bool, optional): Undersample if True. Defaults to False.

        Returns:
            score_list (list): Scores for each metric
        """

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
                
                X_train_fold, X_test_fold, y_train_fold, y_test_fold= self.random_spilt(rate=1/self.test_clusters, modify=modify) 
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
            # OoD by K-means
            test_mask_list, dataset_list = self.dataset_split(test_mask, modify=modify)
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

            pred_arr, y_arr = algo.predicter(self.feature_arr)
            show_result_map(result_values=y_arr, mask=self.common_mask, deposit_mask=getDepositMask(self.path), test_mask=test_mask_list[len(test_mask_list)-1])
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
        auc_std, f_std = np.mean(std_arr[:,0]), np.mean(std_arr[:,1])
        y = list(y) + [auc_std, f_std]
              
        return y