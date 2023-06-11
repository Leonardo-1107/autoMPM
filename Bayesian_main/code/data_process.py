from utils import *

if __name__ == '__main__':
    # For datasets preprocess, except Washington
    preprocess_all_data(output_dir='./data_benchmark', target_name='Au', label_filter=True)

    # Specially for Washington
    preprocess_data_interpolate()
    preprocess_Nova_data(data_dir='./dataset/NovaScotia2')
    preprocess_data(
        data_dir=f'./dataset/nefb_fb_hlc_cir', 
        feature_list=['As', 'B', 'Ca', 'Co', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Y'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask1.tif', 
        target_name='Cr', 
        label_path_list=['shape/nefb_fb_hlc_cir_Deposites.shp', 'shape/nefb_fb_hlc_cir_deposit_Au_quarzt.shp', 'shape/nefb_fb_hlc_cir_deposit_Ploymetallic_vein.shp'], 
        output_path=f'./data/nefb_fb_hlc_cir_cr.pkl',
        label_filter=True
        )

    preprocess_data(
        data_dir=f'./dataset/tok_lad_scsr_ahc', 
        feature_list=['B', 'Ca1', 'Co1', 'Cu', 'Fe', 'Mg', 'Mn', 'Ni', 'Pb', 'Sc', 'Sr', 'V', 'Y', 'Zn'], 
        feature_prefix='raster/', 
        feature_suffix='.tif', 
        mask='raster/mask.tif', 
        target_name='Cr', 
        label_path_list=['shape/tok_lad_scsr_ahc_Basaltic_Cu_Au.shp','shape/tok_lad_scsr_ahc_porphyry_Cu_Au.shp', 'tok_lad_scsr_ahc_deposites.shp', 'tok_lad_scsr_ahc_Placer_Au.shp'], 
        output_path=f'./data/tok_lad_scsr_ahc_cr.pkl',
        label_filter=True
        )

    preprocess_data(
    data_dir='./dataset/North Idaho', 
    feature_list=['ba', 'ca', 'cr', 'cu', 'fe', 'la', 'mg', 'mn', 'ni', 'pb', 'sr', 'ti', 'v', 'y', 'zr'], 
    feature_prefix='Raster/Geochemistry/', 
    feature_suffix='', 
    mask='Raster/Geochemistry/pb', 
    target_name='Ag', 
    label_path_list=['Shapefiles/Ag.shp'], #, 'Shapefiles/mineral_deposit.shp'
    output_path=f'./data/North_Idaho_ag.pkl',
    label_filter=False
    )