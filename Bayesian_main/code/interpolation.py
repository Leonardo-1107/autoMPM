import pykrige
from scipy.interpolate import interp2d
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pylab as pylab
from multiprocessing import Pool

# Version 1.0 - measured by MSE in one step
# Version 1.1 - measured by performance of our system in 3 steps

class interp_kriging:
    def __init__(self):
        return
    
    def interp(self, x, y, data, x_max, y_max, method):
        okri = pykrige.ok.OrdinaryKriging(x, y, data, variogram_model = method, nlags = 50)
        interp_arr2d, _ = okri.execute("grid", np.arange(x_max), np.arange(y_max))
        return interp_arr2d

class interp_opt:
    def __init__(self, proportion=0.2, algo_list=None, num_processes=4):
        self.proportion = proportion
        self.kriging_method = ["linear", "gaussian", "exponential", "hole-effect"]
        self.interp2d_method = ['linear', 'cubic', 'quintic']
        if algo_list is None:
            self.algo_list = [interp_kriging, interp2d]
        else:
            self.algo_list = algo_list
        self.num_processes = num_processes

    def data_split(self, x, y, data):
        assert len(x) == len(y) and len(x) == len(data)
        num = len(x)
        test = np.random.choice(num, int(num * self.proportion))
        test_mask = np.zeros_like(x).astype(bool)
        test_mask[test] = True
        test_set = (x[test_mask], y[test_mask], data[test_mask])
        train_set = (x[~test_mask], y[~test_mask], data[~test_mask])
        return train_set, test_set

    def optimize_one(self, interp_algo, train_set, x_max, y_max, method):
        if interp_algo == interp_kriging:
            interp_method = interp_algo()
            result = interp_method.interp(train_set[0], train_set[1], train_set[2], float(x_max), float(y_max), method).T
            error = mean_squared_error(result, train_set[2])
        else:
            interp_method = interp_algo(train_set[0], train_set[1], train_set[2], kind=method)
            result = train_set[2].copy()
            for i in range(x_max):
                for j in range(y_max):
                    if result[i, j] == 0:
                        result[i, j] = interp_method(i, j)
            error = mean_squared_error(result, train_set[2])
        return result, method, error

    def optimize(self, x, y, data, x_max, y_max):
        train_set, test_set = self.data_split(x, y, data)
        best_error = float('inf')  # Initialize with a high error
        best_result = None
        best_method = None

        with Pool(self.num_processes) as pool:
            results = []
            for interp_algo in self.algo_list:
                for method in (self.kriging_method if interp_algo == interp_kriging else self.interp2d_method):
                    results.append(pool.apply_async(self.optimize_one, args=(interp_algo, train_set, x_max, y_max, method)))

            for result in results:
                result, method, error = result.get()
                if error < best_error:
                    best_error = error
                    best_result = result
                    best_method = method

        return best_result, best_method, best_error

if __name__ == '__main__':

    
    # Example
    import rasterio
    import geopandas

    data_dir = 'Bayesian_main/dataset/Washington'
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
    # Data augment
    label_arr = label_arr2d[mask]
    
    geochemistry = geopandas.read_file(data_dir+'/shapefile/Geochemistry.shp')   
    feature_list = ['B', 'Ca', 'Cu', 'Fe', 'Mg', 'Ni']
    
    feature = 'Mg'
    feature_dict = {}
    size = mask_ds.index(mask_ds.bounds.right, mask_ds.bounds.bottom)
    global feature_data
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
    z = geochemistry[feature].values
    
    interpOPT = interp_opt()
    result = interpOPT.optimize(x_geo, y_geo, z, x_max, y_max)
    
    