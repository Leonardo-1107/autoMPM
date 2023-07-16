from utils import *

import os
os.chdir('Bayesian_main')
 

if __name__ == '__main__':
    # For datasets preprocess, except Washington
    # preprocess_all_data(output_dir='./data_benchmark', target_name='Au', label_filter=True)

    # Specially for Washington
    preprocess_data_interpolate(method='linear')
    # preprocess_Nova_data(data_dir='./dataset/NovaScotia2')
    data_dir = './dataset'
    
    # preprocess_data(
    #     data_dir=f'{data_dir}/bm_lis_go_sesrp', 
    #     feature_list=['ag_ppm', 'as', 'be_ppm', 'ca', 'co', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'ti'], 
    #     feature_prefix='raster/', 
    #     feature_suffix='', 
    #     mask='raster/mask.tif', 
    #     target_name='', 
    #     label_path_list=['shapefile/bm_lis_go_quartzveinsAu.shp'], 
    #     output_path='./data_benchmark/other/bm_lis_go_sesrp.pkl',
    #     label_filter=False
    #     )
