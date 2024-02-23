import pandas as pd
import numpy as np
import rasterio
import pickle
from pre_process_functions import *


class Preprocess:
    def __init__(self, geology_file, lake_sediment_file, mask_file):
        self.geology_file = geology_file
        self.lakesediment_file = lake_sediment_file
        self.basemap_file = mask_file

    def contact_boundary_extraction(self):
        """
        Vector filtering and contact boundary extraction based on attributes
        """
        layer_halifax = vector_filter(self.geology_file, 'LITHOLGY', ['Halifax Formation'])
        layer_godenville = vector_filter(self.geology_file, 'LITHOLGY', ['Godenville Formation'])
        intersection_boundary(layer_halifax, layer_godenville, 'intersection')
        vector_to_raster_mask('intersection/intersection.shp', self.basemap_file, 'intersection.tif')
        euclidean_distance_analysis('intersection.tif', self.basemap_file, 'intersection_distance.tif')

    def element_extraction(self, element_list=["COPPER", "LEAD", "ZINC", "ARSENIC"]):
        """
        Pre-process the element for the geology fike
        """
        for name in element_list:
            ordinary_kriging_interpolation(self.lakesediment_file, self.basemap_file, f'{name}_ok.tif', name)
    
    
    def make_labels_files(self, input_layer_path, buffer_distance, output_layer_path):
        """
        Make the labels
        """
        buffer_ds = vector_buffering(input_layer_path, buffer_distance)
        vector_to_raster_mask(buffer_ds, self.basemap_file, output_layer_path)
    
    def make_dataset(self, feature_list, mask_file, label_file, output_name = 'NewNova'):
        """
        Integrate into a whole dataset        
        """
        input_layers_array, input_layers_pd, dimension, ds_mask = generate_input_layers_data(
            input_layers_path_list=feature_list,
            input_mask_path=mask_file,
            input_label_path=label_file,
        )

        X, y, mask_t_f, X_scale_np = generate_model_dataset(input_layers_pd, ds_mask, 2)

        mask_ds = rasterio.open(label_file)
        mask_data = mask_ds.read(1)
        mask = mask_data <255
        common_mask = mask

        dataset = (X, np.array([y, y]), common_mask, feature_list)
        with open(output_name, 'wb') as f:
            pickle.dump(dataset, f)


# Example of how to use the Preprocess class
if __name__ == "__main__":


    # Put the files into the same fold
    preprocessor = Preprocess('geology.shp', 'lakesediment.shp', 'basemap.tif')

    # To generate the original files
    preprocessor.contact_boundary_extraction()
    preprocessor.element_extraction()

    # make the labels
    preprocessor.make_labels_files('deposit.shp', 3000, 'label_buffer.tif')

    # finally output the packed dataset
    preprocessor.make_dataset(
        feature_list=['anticline_distance.tif', 'intersection_distance.tif', 'as_ok.tif', 'cu_ok.tif', 'pb_ok.tif', 'zn_ok.tif'],
        mask_file='basemap.tif',
        label_file='label_buffer.tif',
        output_name='NewNova')





