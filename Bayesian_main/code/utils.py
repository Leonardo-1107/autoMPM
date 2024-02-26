import rasterio
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import heapq
from queue import Queue as pyQueue
import scipy.stats as ss
from scipy.interpolate import interp2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc, roc_curve, make_scorer, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
                
# import pykrige
import time
import os
import sys
# from metric import Feature_Filter
from interpolation import interp_opt

"""
The early stage data preprocess and some plot functions.

"""

def preprocess_data(data_dir='./dataset/nefb_fb_hlc_cir', feature_list=['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], feature_prefix='', feature_suffix='.tif', mask='raster/mask1.tif', target_name='Au', label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], augment=True, label_filter=True, feature_filter=False, output_path='./data/nefb_fb_hlc_cir.pkl'):
    """Preprocess the dataset from raster files and shapefiles into feature, label and mask data

    Args:
        data_dir (str, optional): The directory of raw data. Defaults to '../../dataset/nefb_fb_hlc_cir'.
        feature_list (list, optional): The list of features to be used. Defaults to ['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'].
        feature_prefix (str, optional): The prefix before the feature name in the path of feature raw data. Defaults to ''.
        feature_suffix (str, optional): The suffix behind the feature name in the path of feature raw data. Defaults to '.tif'.
        mask (str, optional): The path of mask raw data. Defaults to 'raster/mask1.tif'.
        target_name (str, optional): The name of target. Defaults to 'Au'.
        label_path_list (list, optional): The list of path of label raw data. Defaults to ['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'].
        augment (bool, optional): Whether to perform data augment operations. Defaults to True.
        label_filter (bool, optional): Whether to fileter the label raw data before process. Defaults to True.
        feature_filter (bool, optional): Whether to fileter the raw features before process instead of using feature list. Defaults to False.
        output_path (str, optional): The path of output data files. Defaults to '../data/nefb_fb_hlc_cir.pkl'.

    Returns:
        Array: The array of samples' feature
        Array: The array of samples' label
        Array: The array of mask
        list: The list of features' name
    """
    
    # Load feature raw data
    feature_dict = {}
    for feature in feature_list:
        rst = rasterio.open(data_dir+f'/{feature_prefix}{feature}{feature_suffix}')
        feature_dict[feature] = rst.read(1)
        
    # Load mask raw data and preprocess
    mask_ds = rasterio.open(data_dir+f'/{mask}')
    mask_data = mask_ds.read(1)
    mask = make_mask(data_dir, mask_data)
    
    # More features added and filtered 
    if feature_filter:
        dirs = os.listdir(data_dir + '/TIFs')
        for feature in dirs:
            if 'tif' in feature:
                if 'toline.tif' in feature:
                    continue
                rst = rasterio.open(data_dir + '/TIFs/' + feature).read(1)
                if rst.shape != mask.shape:
                    continue
                feature_list.append(feature)
                feature_dict[feature] = np.array(rst) 

    # Preprocess feature
    feature_arr = np.zeros((mask.sum(),len(feature_list)))
    for i, feature in enumerate(feature_list):
        feature_arr[:, i] = feature_dict[feature][mask]
        
    # Load label raw data
    label_x_list = []
    label_y_list = []
    for path in label_path_list:
        deposite = geopandas.read_file(data_dir+f'/{path}')
        # Whether to filter label raw data
        if label_filter:
            deposite = deposite.dropna(subset='comm_main')
            au_dep = deposite[[target_name in row for row in deposite['comm_main']]]
        else:
            au_dep = deposite
        # Extract the coordinate
        label_x = au_dep.geometry.x.to_numpy()
        label_y = au_dep.geometry.y.to_numpy()

        label_x_list.append(label_x)
        label_y_list.append(label_y)

    # Preprocess label
    x = np.concatenate(label_x_list)
    y = np.concatenate(label_y_list)
    row, col = mask_ds.index(x,y)
    row_np = np.array(row)
    row_np[row_np == mask_data.shape[0]] = 1
    label_arr2d = np.zeros_like(mask_data)
    for x, y in zip(row_np, col):
        label_arr2d[x, y] = 1

    deposite_mask = label_arr2d
    ground_label_arr = label_arr2d[mask]
    label_arr = ground_label_arr
    # Data augment
    if augment:
        label_arr2d = augment_2D(label_arr2d)
        label_arr = label_arr2d[mask]
    
    # feature filtering
    if feature_filter:
        feature_filter_model = Feature_Filter(input_feature_arr=feature_arr)
        feature_arr = feature_filter_model.select_top_features(top_k=20)

    # Pack and save dataset
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


def preprocess_all_data(data_dir='./dataset', output_dir='./data', target_name='Au', label_filter=True, augment=False):
    preprocess_data(
        data_dir=f'{data_dir}/nefb_fb_hlc_cir', 
        feature_list=['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask1.tif', 
        target_name=target_name, 
        label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], 
        output_path=f'{output_dir}/nefb_fb_hlc_cir.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/tok_lad_scsr_ahc', 
        feature_list=['B', 'Ca1', 'Co1', 'Cr1', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Sr', 'V', 'Y', 'Zn'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask.tif', 
        target_name=target_name, 
        label_path_list=['shape/tok_lad_scsr_ahc_Basaltic_Cu_Au.shp','shape/tok_lad_scsr_ahc_porphyry_Cu_Au.shp', 'tok_lad_scsr_ahc_Placer_Au.shp'], 
        output_path=f'{output_dir}/tok_lad_scsr_ahc.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/North Idaho', 
        feature_list=['ba', 'ca', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'sr', 'ti', 'v', 'y', 'zr'], 
        feature_prefix='Raster/Geochemistry/', 
        feature_suffix='', 
        mask='Raster/Geochemistry/pb', 
        target_name=target_name, 
        label_path_list=['Shapefiles/Au.shp'], #, 'Shapefiles/mineral_deposit.shp'
        output_path=f'{output_dir}/North_Idaho.pkl',
        label_filter=False,
        augment=augment
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/bm_lis_go_sesrp', 
        feature_list=['ag_ppm', 'as', 'be_ppm', 'ca', 'co', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'ti'], 
        feature_prefix='raster/', 
        feature_suffix='', 
        mask='raster/mask.tif', 
        target_name=target_name, 
        label_path_list=['shapefile/bm_lis_go_quartzveinsAu.shp'], 
        output_path=f'{output_dir}/bm_lis_go_sesrp.pkl',
        label_filter=label_filter,
        augment=augment
        )
    
    
def preprocess_data_interpolate(data_dir='./dataset/Washington', augment:bool = True, method = 'kriging', feature_filter = False):
    """
    Convert point data to raster data by interpolation

    Args:
        data_dir (str, optional): The directory of raw data. 
        augment (bool, optional): Whether to perform data augment operations. Defaults to True.
        method (str, optional): The method for interpolation
        feature_filter (bool, optional): Whether to fileter the raw features before process instead of using feature list. Defaults to False.
        
    Returns:
        Array: The array of samples' feature
        Array: The array of samples' label
        Array: The array of mask
        list: The list of features' name
    """
    
    mask_ds = rasterio.open(data_dir+'/shapefile/mask1.tif')
    mask_data = mask_ds.read(1)
    mask = mask_data == 1
    
    au = geopandas.read_file(data_dir+'/shapefile/Au.shp')
    x = au.geometry.x.to_numpy()
    y = au.geometry.y.to_numpy()
    row, col = mask_ds.index(x,y)

    row_np = np.array(row)
    row_np[np.array(row) == mask_data.shape[0]] = 1
    label_arr2d = np.zeros_like(mask_data)
    for x, y in zip(row_np, col):
        label_arr2d[x, y] = 1
    
    deposite_mask = label_arr2d
    ground_label_arr = label_arr2d[mask]
    if augment:
        label_arr2d = augment_2D(label_arr2d)
    label_arr = label_arr2d[mask]
    
    geochemistry = geopandas.read_file(data_dir+'/shapefile/Geochemistry.shp')   
    feature_list = ['B', 'Ca', 'Cu', 'Fe', 'Mg', 'Ni']
    feature_dict = {}
    size = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)
    for feature in feature_list:
        feature_data = np.zeros(size)
        for i, row in geochemistry.iterrows():
            x = row.geometry.x
            y = row.geometry.y
            x, y = mask_ds.index(x, y)
            data = row[feature]
            if data < 1e-8:
                data = 1e-8
            feature_data[x, y] = data
            feature_dict[feature] = feature_data
        
    x_geo, y_geo = geochemistry.geometry.x.values, geochemistry.geometry.y.values
    x_max, y_max = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)

    # Interpolation to transfer shapfiles to rater form
    for feature in feature_list:
        print(f'Processing {feature}')
        z = geochemistry[feature].values

        # interp optimization
        interpOPT = interp_opt()
        result = interpOPT.optimize(x_geo, y_geo, z, x_max, y_max)

        feature_dict[feature] = result
            
    feature_arr2d_dict = feature_dict.copy()
    feature_arr = np.zeros((mask.sum(),len(feature_list)))
    for idx in range(len(feature_list)):
        feature_arr[:,idx] = feature_arr2d_dict[feature_list[idx]][mask]

    if feature_filter:
        feature_filter_model = Feature_Filter(input_feature_arr=feature_arr)
        feature_arr = feature_filter_model.select_top_features(top_k=20)

    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(f'./data/Washington_{method}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

def preprocess_Nova_data(data_dir, feature_prefix='', feature_suffix='.npy', mask_dir='Mask.npy', label_path_list=['Target.npy'], augment=False, output_path = './data_benchmark/Nova.pkl'):
    # Process the NovaScotia2 Data
    feature_list = ['Godenville_Formation_Buffer', 'Anticline_Buffer', 'As', 'Li', 'Pb', 'F', 'Cu', 'W', 'Zn']
    feature_dict = {}
    for feature in feature_list:
        rst = np.load(data_dir+f'/{feature_prefix}{feature}{feature_suffix}')
        feature_dict[feature] = rst
        
    
    # Load mask data and preprocess
    mask = np.load(data_dir+ '/' +mask_dir).astype(np.int64)
    mask = make_mask(data_dir, mask_data=mask, show=True)

    # Preprocess features
    feature_arr = np.zeros((mask.sum(), len(feature_list)))
    for i, feature in enumerate(feature_list):
        feature_arr[:, i] = feature_dict[feature][mask]
    
    # Load the target ID
    label_arr = np.zeros(shape=(feature_arr.shape[0], ))
    
    for path in label_path_list:
        depositMask = (np.load(data_dir + '/' + path) > 0 )
        ground_label_arr = depositMask[mask]
        label_arr = ground_label_arr

        if augment:
            label_arr2d = augment_2D(depositMask) 
            label_arr = label_arr2d[mask]
    
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, depositMask)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset


def make_mask(data_dir, mask_data, show =False):

    if 'nefb' in data_dir or 'tok' in data_dir or 'Washington' in data_dir:
        mask = mask_data != 0
    
    elif 'bm' in data_dir:
        mask = mask_data == 1

    elif 'North' in data_dir:
        mask = (mask_data > -1)
    
    else:
        mask = mask_data != 0

    if show:
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()
        name = data_dir.replace('/','')
        plt.title("Mask")
        # plt.savefig(f'./backup/mask_{name}.png')

    return mask

def augment_2D(array, wide_mode = False):
    """
    For data augment function. Assign the 3*3 blocks around the sites to be labeled.
    """
    new = array.copy()
    a = np.where(array == 1)
    x, y = a[0], a[1]
    aug_touple = [(-1,-1),(-1,1),(1,-1),(1,1),(0,1),(0,-1),(1,0),(-1,0)]
    print(array.sum())
    if wide_mode:
        aug_touple = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2),  (0, -1),  (0, 0),  (0, 1),  (0, 2),
            (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
            (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2),
        ]

    for idx in range(len(x)):
        for m,n in aug_touple:
            newx = x[idx] + m
            newy = y[idx] + n
            
            if (0< newx and newx < array.shape[0]) and (0< newy and newy < array.shape[1]):
                new[newx][newy] = 1
    return new


def plot_roc(fpr, tpr, index, scat=False, save=True):
    """
        plot ROC curve
    """
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'split set {index}, ROC area = {roc_auc:.2f}', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    if scat:
        plt.scatter(fpr, tpr)
    plt.title("ROC curve for mineral prediction")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    if save:
        plt.grid(alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Bayesian_main/run/figs/roc.png')
    else:
        plt.show()    


def plot_PR(y_test_fold, y_arr, index):
    """
    plot Precision-Recall curve
    """
    prec, recall, _ = precision_recall_curve(y_test_fold, y_arr)
    non_zero_indices = np.logical_and(prec != 0, recall != 0)
    f1_scores = 2 * (prec[non_zero_indices] * recall[non_zero_indices]) / (prec[non_zero_indices] + recall[non_zero_indices])
    max_f1_score = np.max(f1_scores)
    max_f1_score_index = np.argmax(f1_scores)
    plt.plot(recall, prec, label = f'split set: {index}, Max F1: {max_f1_score:.2f}')
    plt.scatter(recall[max_f1_score_index], prec[max_f1_score_index], c='red', marker='o')
    plt.legend()
    plt.grid(alpha=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve for mineral prediction")
    plt.tight_layout()

    plt.savefig('Bayesian_main/run/figs/precision-recall.png', dpi=300)

def get_PR_curve(pred_list):
    plt.figure()
    for i in range(len(pred_list)):
            y_test_fold, y_arr = pred_list[i]
            plot_PR(y_test_fold, y_arr, i+1)
    
def get_ROC_curve(pred_list):
    plt.figure()

    for i in range(len(pred_list)):
        y_test_fold, y_arr = pred_list[i]
        fpr, tpr, thersholds = roc_curve(y_test_fold, y_arr)
        plot_roc(fpr, tpr, i+1)

def get_confusion_matrix(cfm_list, clusters):
    """
    plot the confusion matrix
    """
    cols = clusters
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(10, 5))
    for i, plt_image in enumerate(cfm_list):
        index2 = i 
        axes[index2].matshow(plt_image, cmap=plt.get_cmap('Blues'), alpha=0.5)
        axes[index2].set_title(f"split set {i+1}")

        # Add labels to each cell
        for j in range(plt_image.shape[0]):
            for k in range(plt_image.shape[1]):
                text = plt_image[j, k]
                axes[index2].annotate(text, xy=(k, j), ha='center', va='center', 
                                      color='black',  weight='heavy', 
                                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1.5, alpha=0.8))
                
                # Add labels for y_pred
                axes[index2].set_xticks(np.arange(plt_image.shape[1]))
                axes[index2].set_xticklabels(np.arange(plt_image.shape[1]))
                axes[index2].set_xlabel("Prediction")

                # Add labels for y_true
                axes[index2].set_yticks(np.arange(plt_image.shape[0]))
                axes[index2].set_yticklabels(np.arange(plt_image.shape[0]))
                axes[index2].set_ylabel("Label")

    fig.tight_layout()
    plt.savefig('./Bayesian_main/run/figs/cfm.png')


def plot_split_standard(common_mask, label_arr, test_mask, save_path=None):
    """
        Plot to demonstrate data split
    """
    plt.figure(dpi=300)
    x, y = common_mask.nonzero()
    positive_x = x[label_arr.astype(bool)]
    positive_y = y[label_arr.astype(bool)]
    test_x, test_y = test_mask.nonzero()
    plt.scatter(x, y)

    plt.scatter(test_x, test_y, color='red')
    plt.scatter(positive_x, positive_y, color='gold')
    plt.legend(['Valid Region', 'Test-set', 'Positive samples'])
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig('./run/spilt_standard.png')

def show_result_map(result_values, mask, deposit_mask, test_mask=None, index=0, clusters=4):
    cols = int((clusters + 0.5) / 2)
    # cols = 1
    if index == 1:
        plt.figure(dpi=600)
        plt.subplots(2, cols, figsize=(15, 15), sharex=True, sharey=False)

    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype="float")

    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i] * 100

    result_array[~mask] = np.nan

    plt.subplot(2, cols, index)
    plt.imshow(result_array, cmap='viridis')
    plt.rcParams['font.size'] = 18

    # Plot target points with improved style
    plt.scatter(dep_XArray, dep_YArray, c='red', s=20, alpha=0.5, label=f"Target (split set {index})")


    # Add a legend
    plt.legend(loc='upper left',fontsize=18)
    cbar = plt.colorbar(shrink=0.9, aspect=30, pad=0.02)
    cbar.ax.set_ylabel('Prediction', fontsize=20)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    # Adjust subplot spacing
    plt.tight_layout()
    plt.gca().set_facecolor('lightgray')

    # Save the figure
    plt.savefig('./Bayesian_main/run/figs/result_map.png', dpi=300)
    t = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(f"\t--- {t} New feature map Saved\n")


def criterion_loss(mode, algo, X_val_fold, y_val_fold, y_train_fold = None):
    
    # the loss applied in training process
    class_weights = compute_class_weight('balanced', classes=np.unique(y_val_fold), y=y_val_fold)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_val_fold), class_weights)}
    
    def weighted_cross_entropy(y_true, y_pred, weight = 'balanced', epsilon = 1e-7):
        sample_weights = compute_sample_weight(class_weight_dict, y_true)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(sample_weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return loss
                
    weighted_ce_scorer = make_scorer(weighted_cross_entropy, greater_is_better=False)
    return weighted_ce_scorer(algo, X_val_fold, y_val_fold)

def load_test_mask(name):
    if 'ag' in name.lower():
        return np.load('./temp/Ag_mask.npy')

    elif 'cu' in name.lower():
        return np.load('./temp/Cu_mask.npy')
    
    elif 'nova' in name.lower():
        return np.load('./temp/New_Nova_mask.npy')
    
    else:
        return np.load('./temp/Au_mask.npy')


def get_feature_importance_by_shap(file_path, per=15):
    '''
    Get the feature importance by shap values
    file_path: the location where the shap_values are saved
    per: the percentage of features to be filtered by shap values
    return: the indices of features to be selected
    '''
    shap_values = np.load(file_path, allow_pickle=True)
    shap_values = [shap_values[0], shap_values[1]]


     
    feature_importance = np.abs(shap_values).mean(0)
    
    threshold = np.percentile(feature_importance, per)
    selected_features = feature_importance > threshold
    top_per_indices = np.argsort(np.sum(selected_features, axis=0))[::-1][:int((1-per/100)*selected_features.shape[1])]
    return top_per_indices,shap_values


import shap
def get_shap(algo, feature_array, mask = None):
    '''
    output the shapely values by the python shap package
    mask: to filter the input feature array. example: indices_of_all_positive
    '''

    if mask is not None:
        feature_array = feature_array[mask]

    try:   
        explainer = shap.Explainer(algo)
        shap_values = explainer.shap_values(feature_array)    
        values = shap_values[1]
    except:
        explainer = shap.Explainer(algo.predict,feature_array)
        shap_values = explainer(feature_array)
        values = np.array(shap_values.values)

    return values


def shap_draw(shap_values, feature_array, feature_names_list, name = 'default_shap_name'):
    '''
    draw the shap value map
    '''

    plt.figure()
    shap.summary_plot(shap_values, feature_array, feature_names=feature_names_list)
    plt.savefig(f'{name}.png')
    
