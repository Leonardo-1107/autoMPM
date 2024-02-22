from utils import *

import os
os.chdir('Bayesian_main')
 

if __name__ == '__main__':

    data_dir = './dataset'
    output_dir = './data'

    '''
        For the data pre-process, all the processed datasets will be 
        stored as a .pkl file in the output dir 
    '''
    preprocess_all_data(output_dir='./data_benchmark', target_name='Au', label_filter=True)
    preprocess_data_interpolate(method='linear')
    preprocess_Nova_data(data_dir='./dataset/NovaScotia2', output_path=output_dir+'/Nova.pkl')



