import rasterio
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import heapq
import scipy.stats as ss
from scipy.interpolate import interp2d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc
import pykrige
import time
import os
import sys

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
        dirs = os.listdir(data_dir + '/Shapefiles')
        for feature in dirs:
            if 'tif' in feature:
                if 'toline.tif' in feature:
                    continue
                rst = rasterio.open(data_dir + '/Shapefiles/' + feature).read(1)
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
    
    # Data augment
    if augment:
        label_arr2d = augment_2D(label_arr2d)
        label_arr = label_arr2d[mask]
    
    # plt.savefig(f'./backup/compare_aug/{time.time()}com.png')
    if feature_filter:
        # Feature filtering by corr
        feature_arr = feature_selecter_corr(feature_arr, ground_label_arr)
        # Feature filtering by weights of RFC
        feature_arr = feature_selecter_algo(feature_arr, label_arr)

    
    # Pack and save dataset
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


def preprocess_all_data(data_dir='./dataset', output_dir='./data', target_name='Au', label_filter=True):
    preprocess_data(
        data_dir=f'{data_dir}/nefb_fb_hlc_cir', 
        feature_list=['As', 'B', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask1.tif', 
        target_name=target_name, 
        label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], 
        output_path=f'{output_dir}/nefb_fb_hlc_cir.pkl',
        label_filter=label_filter
        )
    
    preprocess_data(
        data_dir=f'{data_dir}/tok_lad_scsr_ahc', 
        feature_list=['B', 'Ca1', 'Co1', 'Cr1', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Sr', 'V', 'Y', 'Zn'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask.tif', 
        target_name=target_name, 
        label_path_list=['shape/tok_lad_scsr_ahc_Basaltic_Cu_Au.shp','shape/tok_lad_scsr_ahc_porphyry_Cu_Au.shp', 'tok_lad_scsr_ahc_deposites.shp', 'tok_lad_scsr_ahc_Placer_Au.shp'], 
        output_path=f'{output_dir}/tok_lad_scsr_ahc.pkl',
        label_filter=label_filter
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
        label_filter=False
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
        label_filter=label_filter
        )
    
    
def preprocess_data_interpolate(data_dir='Washington', augment:bool = True):
    """
    Convert point data to raster data by interpolation

    Args:
        data_dir (str, optional): The directory of raw data. 
        augment (bool, optional): Whether to perform data augment operations. Defaults to True.


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

    # Feature filtering by corr
    feature_arr = feature_selecter_corr(feature_arr, ground_label_arr)
    # Feature filtering by weights of RFC
    feature_arr = feature_selecter_algo(feature_arr, label_arr)
    
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
        method = 'kriging'
        
        if method == 'kriging':
            
            OK = pykrige.OrdinaryKriging(
                x_geo,
                y_geo,
                z,
                variogram_model="gaussian",  
            )
            gridX = np.linspace(np.min(x_geo), np.max(x_geo), x_max)
            gridY = np.linspace(np.min(y_geo), np.max(y_geo), y_max)
            feature_dict[feature], _ = OK.execute("grid", gridX, gridY)
            feature_dict[feature] = feature_dict[feature].T
            print('Feature checking:  ', feature_dict[feature].shape == mask.shape)
        else:
            f = interp2d(x_geo, y_geo, z, kind=method)
            for x in range(x_max):
                for y in range(y_max):
                    if feature_dict[feature][x, y] == 0:
                        feature_dict[feature][x, y] = f(x, y)
            
    feature_arr2d_dict = feature_dict.copy()
    feature_arr = np.zeros((mask.sum(),len(feature_list)))
    for idx in range(len(feature_list)):
        feature_arr[:,idx] = feature_arr2d_dict[feature_list[idx]][mask]
     
    dataset = (feature_arr, np.array([ground_label_arr, label_arr]), mask, deposite_mask)
    with open(f'./data/Washington_{method}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

def make_mask(data_dir, mask_data, show =False):

    if 'nefb' in data_dir or 'tok' in data_dir or 'Washington' in data_dir:
        mask = mask_data != 0
    
    if 'bm_lis' in data_dir:
        mask = (mask_data < 200)

    if 'North' in data_dir:
        mask = (mask_data > -1)
    if show:
        plt.figure()
        plt.imshow(mask)
        plt.colorbar()
        name = data_dir.replace('/','')
        plt.savefig(f'./backup/mask_{name}.png')

    return mask

def augment_2D(array):
    """
    For data augment function. Assign the 3*3 blocks around the sites to be labeled.
    """
    new = array.copy()
    a = np.where(array == 1)
    x, y = a[0], a[1]
    aug_touple = [(-1,-1),(-1,1),(1,-1),(1,1),(0,1),(0,-1),(1,0),(-1,0)]
    for idx in range(len(x)):
        for m,n in aug_touple:
            newx = x[idx] + m
            newy = y[idx] + n
            
            if (0< newx and newx < array.shape[0]) and (0< newy and newy < array.shape[1]):
                new[newx][newy] = 1
                 
    return new


def feature_selecter_corr(feature_arr, label_arr):
    """
    Initial screening of features by calculating correlation coefficients.
    """
    corr_list = []
    for i in range(feature_arr.shape[1]): 
        feature_slice = feature_arr[:,i]
        corr = ss.pearsonr(feature_slice, label_arr)
        corr_list.append(corr[0])
    
    threshold = heapq.nlargest(50, corr_list)[49]
    select_list = [x>= threshold for x in corr_list]
    
    selected_feature_arr = []
    for i in range(feature_arr.shape[1]):
        if select_list[i]:
            selected_feature_arr.append(feature_arr[:,i])
    
    return np.array(selected_feature_arr).T
    
def feature_selecter_algo(feature_arr, label_arr, n = 20):
    """
    Fine screening of features by the weights obtained in the random forest algorithm.
    """
    rfc = RandomForestClassifier()
    rfc.fit(feature_arr, label_arr)
    
    weights = rfc.feature_importances_
    threshold = heapq.nlargest(n, weights)[n-1]
    select_list = [x>= threshold for x in weights]
    
    selected_feature_arr = []
    for i in range(feature_arr.shape[1]):
        if select_list[i]:
            selected_feature_arr.append(feature_arr[:,i])
    
    return np.array(selected_feature_arr).T

def getDepositMask(name = 'Washington'):
    """
    Return the mineral sites when drawing the feature map
    """

    if name == 'Bayesian_main/data/NovaScotia.pkl':
        return np.load("Bayesian_main/dataset/NovaScotia/deposit.npy")
    
    if name == 'Bayesian_main/data/NovaScotia2.pkl':
        return np.load("Bayesian_main/dataset/NovaScotia2/Target.npy")
    
    if 'North' in name:
        data_dir = 'Bayesian_main/dataset/North Idaho'
        mask_ds = rasterio.open(data_dir+'/Shapefiles/mask.tif')
        mask_data = mask_ds.read(1)
        au = geopandas.read_file(data_dir+'/Shapefiles/Au.shp')
        
        
    if 'Washington' in name:
        mask_ds = rasterio.open('Bayesian_main/dataset/Washington/shapefile/mask1.tif')
        mask_data = mask_ds.read(1)
        au = geopandas.read_file('Bayesian_main/dataset/Washington/shapefile/Au.shp')
    
    if 'bm_lis' in name:
        mask_ds = rasterio.open('Bayesian_main/dataset/bm_lis_go_sesrp/raster/mask.tif')
        mask_data = mask_ds.read(1)
        au = geopandas.read_file('Bayesian_main/dataset/bm_lis_go_sesrp/shapefile/bm_lis_go_quartzveinsAu.shp')

    if 'nefb' in name:
        mask_ds = rasterio.open('Bayesian_main/dataset/nefb_fb_hlc_cir/raster/mask1.tif')
        mask_data = mask_ds.read(1)
        au = geopandas.read_file('Bayesian_main/dataset/nefb_fb_hlc_cir/shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp')
    
    if 'tok_lad' in name:
        mask_ds = rasterio.open('Bayesian_main/dataset/tok_lad_scsr_ahc/raster/mask.tif')
        mask_data = mask_ds.read(1)
        au = geopandas.read_file('Bayesian_main/dataset/tok_lad_scsr_ahc/tok_lad_scsr_ahc_deposites.shp')
        #  Bayesian_main/dataset/tok_lad_scsr_ahc/shape/tok_lad_scsr_ahc_porphyry_Cu_Au.shp

    x = au.geometry.x.to_numpy()
    y = au.geometry.y.to_numpy()
    row, col = mask_ds.index(x,y)

    row_np = np.array(row)
    row_np[np.array(row) == mask_data.shape[0]] = 1
    label_arr2d = np.zeros_like(mask_data)

    for x, y in zip(row_np, col):
        label_arr2d[x, y] = 1

    return label_arr2d

def plot_roc(fpr, tpr, index, scat=False, save=True):
    """
        plot ROC curve
    """
    # plt.figure()
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC area = {0:.2f}'.format(roc_auc), lw=2)
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
        plt.savefig(f'Bayesian_main/run/{index}_roc.png')
    else:
        plt.show()    

def plot_PR(y_test_fold, y_arr):
    """
        plot Precision-Recall curve
    """
    prec, recall, _ = precision_recall_curve(y_test_fold, y_arr)
    plt.plot(recall, prec)

    plt.grid(alpha=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig('Bayesian_main/run/precision-recall.png', dpi=300)

def plot_split_standard(common_mask, label_arr, test_mask, save_path=None):

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

def show_result_map(result_values, mask, deposit_mask, test_mask = None):
    # if test_mask == None:
    #     test_mask = mask
    plt.figure(dpi=300)
    validYArray, validXArray = np.where(mask > 0)
    dep_YArray, dep_XArray = np.where(np.logical_and(deposit_mask, test_mask) == 1)
    result_array = np.zeros_like(deposit_mask, dtype = "float")

    
    for i in range(len(validYArray)):
        if test_mask[validYArray[i], validXArray[i]] > 0:
            result_array[validYArray[i], validXArray[i]] = result_values[i]*100
    
    result_array[~mask] = np.nan
    
    # pylab.imshow(new_mask, cmap='spring')
    pylab.imshow(result_array, cmap='cividis')
    pylab.colorbar(label="0/1", orientation="vertical")

    pylab.scatter(dep_XArray, dep_YArray, c='r', s=5, alpha=0.8, label = "Target")
    plt.legend(fontsize = 14)
    pylab.savefig('result_map.png')
    t = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(f"\t--- {t} New feature map Saved\n")

if __name__=="__main__":
    
    # For datasets preprocess, except Washington
    preprocess_all_data(output_dir='./data_benchmark', target_name='Au', label_filter=True)
    # Specially for Washington
    # preprocess_data_interpolate()
