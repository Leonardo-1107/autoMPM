import pandas as pd
import numpy as np
from osgeo import gdal, ogr
import geopandas as gpd
import pykrige
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


def vector_filter(input_layer_path, field, feature_list):
    input_layer_gdf = gpd.read_file(input_layer_path)
    select_index = input_layer_gdf[field].isin(feature_list)
    select_layer = input_layer_gdf[select_index]
    return select_layer

def intersection_boundary(gdf_layer_1, gdf_layer_2, output_layer_path):
    boundary_1 = gdf_layer_1.boundary.unary_union
    boundary_2 = gdf_layer_2.boundary.unary_union
    intersection_boundary = boundary_1.intersection(boundary_2)
    intersection_boundary_gdf = gpd.GeoSeries(intersection_boundary)
    intersection_boundary_gdf = gpd.GeoDataFrame(geometry=intersection_boundary_gdf, crs=gdf_layer_1.crs)
    intersection_boundary_gdf.to_file(output_layer_path)

def vector_to_raster_mask(input_layer_path, input_mask_path, output_layer_path, field=''):
    ds = gdal.Open(input_mask_path)
    geo_transform = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds1 = driver.Open(input_layer_path, 1)
    lyr = ds1.GetLayer()

    target_ds = gdal.GetDriverByName('GTiff').Create(output_layer_path, xsize=cols, ysize=rows, bands=1,
                                                        eType=gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ds.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    if field == '':
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
    else:
        options = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(target_ds, [1], lyr, None, options=options)

def euclidean_distance_analysis(input_layer_path, input_mask_path, output_layer_path):
    ds1 = gdal.Open(input_layer_path)
    band1 = ds1.GetRasterBand(1)
    layer_np = band1.ReadAsArray()

    ds2 = gdal.Open(input_mask_path)
    band2 = ds2.GetRasterBand(1)
    mask_np = band2.ReadAsArray()

    row, column = mask_np.shape

    mask_bool = get_mask_boolean(mask_np, band2.GetNoDataValue())
    row_index, column_index = np.where(mask_bool)

    layer_dis_np = np.zeros((row, column))

    for i, j in zip(row_index, column_index):
        k = 0
        while True:
            layer_np_pad = np.pad(layer_np, ((k, k), (k, k)), 'constant', constant_values=(0, 0))
            layer_np_clip = layer_np_pad[i:i+2*k+1, j:j+2*k+1]
            row_index_clip, column_index_clip = np.where(layer_np_clip == 1)
            if len(row_index_clip) != 0:
                distance_list = []
                m, n = row_index_clip - k, column_index_clip - k
                distance = np.sqrt(m**2 + n**2)
                layer_dis_np[i, j] = distance.min()
                break
            k += 1

    write_array_to_raster(input_mask_path, layer_dis_np, output_layer_path)

def get_mask_boolean(input_mask_array, input_mask_nodata):
    if True in np.isnan(input_mask_array):
        mask_array_boolean = ~np.isnan(input_mask_array)
    else:
        mask_array_boolean = np.where(input_mask_array == input_mask_nodata, False, True)

    return mask_array_boolean

def write_array_to_raster(input_mask_path, output_lyr_array, output_lyr_path_local):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = output_lyr_array.shape

    ds = gdal.Open(input_mask_path)
    mask_info = get_raster_info(ds)

    mask_array_boolean = get_mask_boolean(mask_info["data"], mask_info["nodata"])

    output_lyr_array[~mask_array_boolean] = -99

    dataset = driver.Create(output_lyr_path_local, cols, rows, 1, eType=gdal.GDT_Float32)
    dataset.SetGeoTransform(mask_info["crs"][1])
    dataset.SetProjection(mask_info["crs"][0])
    band = dataset.GetRasterBand(1)
    band.WriteArray(output_lyr_array)
    band.SetNoDataValue(-99)
    band.ComputeStatistics(False)

def get_raster_info(ds):
    gt = ds.GetGeoTransform()
    in_band = ds.GetRasterBand(1)
    x_size = in_band.XSize
    y_size = in_band.YSize
    mask_data = in_band.ReadAsArray()

    data_description = dict(extent=(gt[0], gt[3], gt[0] + gt[1] * ds.RasterXSize,
                                    gt[3] + gt[5] * ds.RasterYSize), raster_size=gt[1],
                            nodata=in_band.GetNoDataValue(), row_column=(y_size, x_size), data=mask_data,
                            crs=[ds.GetProjection(), ds.GetGeoTransform()])
    return data_description

def ordinary_kriging_interpolation(input_layer_path, input_mask_path, output_layer_path, feature):
    input_mask_ds = gdal.Open(input_mask_path)
    mask_geotrans = input_mask_ds.GetGeoTransform()
    mask_xMin = mask_geotrans[0]
    mask_yMax = mask_geotrans[3]
    mask_pixelWidth = mask_geotrans[1]
    mask_pixelHeight = mask_geotrans[5]  # NOTICE: the height is a negative value
    mask_rows = input_mask_ds.RasterYSize
    mask_cols = input_mask_ds.RasterXSize
    mask_xMax = mask_xMin + (mask_cols * mask_pixelWidth)
    mask_yMin = mask_yMax + (mask_rows * mask_pixelHeight)

    input_vector_gdf = gpd.read_file(input_layer_path)
    vector_x_arr1d = input_vector_gdf.geometry.x
    vector_y_arr1d = input_vector_gdf.geometry.y
    grid_x_arr = np.arange(mask_xMin, mask_xMax, mask_pixelWidth)
    grid_y_arr = np.arange(mask_yMin, mask_yMax, -mask_pixelHeight)
    input_vector_arr = np.log(input_vector_gdf[feature].values)

    okri = pykrige.ok.OrdinaryKriging(vector_x_arr1d, vector_y_arr1d, input_vector_arr, variogram_model="exponential", nlags=50)
    interp_arr2d, sigma_arr2d = okri.execute("grid", grid_x_arr, grid_y_arr)

    write_array_to_raster(input_mask_path, np.exp(interp_arr2d[::-1, ]).data, output_layer_path)

def vector_buffering(input_layer_path, buffer_distance) -> ogr.DataSource:

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(input_layer_path, 1)
    lyr = ds.GetLayer()
    
    # Create a temporary shapefile on disk
    temp_output_path = 'buffered_data.shp'
    out_ds = driver.CreateDataSource(temp_output_path)
    out_lyr = out_ds.CreateLayer('buffered_layer', lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)

    feature_define = lyr.GetLayerDefn()
    out_feat = ogr.Feature(feature_define)

    for feature in lyr:
        out_geom = feature.GetGeometryRef()
        buffer_geom = out_geom.Buffer(buffer_distance)
        out_feat.SetGeometry(buffer_geom)
        out_lyr.CreateFeature(out_feat)

    return temp_output_path  # Return the path of the temporary shapefile

def vector_to_raster_mask(input_ds, input_mask_path, output_layer_path, field=''):

    ds = gdal.Open(input_mask_path)
    geo_transform = ds.GetGeoTransform()

    cols = ds.RasterXSize
    rows = ds.RasterYSize

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds1 = driver.Open(input_ds, 1)
    lyr = ds1.GetLayer()

    target_ds = gdal.GetDriverByName('GTiff').Create(output_layer_path, xsize=cols, ysize=rows, bands=1,
                                                    eType=gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ds.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    if field == '':
        gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
    else:
        options = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(target_ds, [1], lyr, None, options=options)

# generate the name of layers via the filenames
def generate_input_layers_data(input_layers_path_list, input_mask_path, input_label_path=''):
    
    input_layers_name_list = extract_layers_name(input_layers_path_list)
    
    if len(input_label_path) != 0:
        input_layers_path_list.extend((input_label_path, input_mask_path))
        input_layers_name_list.extend(('Label', 'Mask'))
    else:
        input_layers_path_list.append(input_mask_path)
        input_layers_name_list.append('Mask')
    input_layers_array = list(gdal.Open(i) for i in input_layers_path_list)
    ds = input_layers_array[-1]
    input_layers_array = np.array(list(map(lambda i: i.GetRasterBand(1).ReadAsArray(), input_layers_array)))
    input_layers_array = np.swapaxes(input_layers_array, 0, 2)
    input_layers_array = np.swapaxes(input_layers_array, 0, 1)
    dimension = input_layers_array.shape
    input_layers_array = input_layers_array.reshape((dimension[0] * dimension[1], dimension[2]))
    input_layers_pd = pd.DataFrame(input_layers_array, columns=input_layers_name_list)

    return input_layers_array, input_layers_pd, dimension, ds

def extract_layers_name(input_layers_path_list):
    
    input_layers = [os.path.split(i)[-1] for i in input_layers_path_list]
    input_layers_name = [os.path.splitext(i)[0] for i in input_layers]

    return input_layers_name


def generate_model_dataset(input_layer_pd, ds_mask):
    mask_no_data = ds_mask.GetRasterBand(1).GetNoDataValue()
    ma_t_f = np.where(input_layer_pd['Mask'] == mask_no_data, False, True)
    
    x_label_mask_pd = input_layer_pd[ma_t_f].copy()
    label_true_index = np.where(x_label_mask_pd['Label'] == 1, True, False)
    x_pd = x_label_mask_pd.drop(labels=['Mask', 'Label'], axis=1)
    x_sc_np = preprocessing.StandardScaler().fit_transform(x_pd)
    x_t_np = x_sc_np[label_true_index, :]
    
    label_false_index = np.where(x_label_mask_pd['Label'] != 1, True, False)
    x_f_np = x_sc_np[label_false_index, :]

    x_train_t = x_t_np
    y_train_t = np.ones((x_t_np.shape[0], 1))

    x_train_f = x_f_np
    y_train_f = np.zeros((x_train_f.shape[0], 1))

    x_tr = np.vstack((x_train_t, x_train_f))
    y_tr = np.vstack((y_train_t, y_train_f))

    x_tr, y_tr = x_y_shuffle(x_tr, y_tr)

    return x_tr, y_tr, ma_t_f, x_sc_np

def x_y_shuffle(x, y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

    return x, y.ravel()