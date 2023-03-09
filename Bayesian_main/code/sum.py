import geopandas as gpd
from matplotlib import pyplot as plt
from osgeo import ogr,osr,os,gdal
import sys 
from shapely.geometry import Point, Polygon, LineString

def get_input_layer(input_layer, output_layer=' '):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(input_layer, 1)
    if ds is None:
        sys.exit('Could not open {0}.'.format(input_layer))
    lyr = ds.GetLayer()
    if os.path.exists(output_layer):
        driver.DeleteDataSource(output_layer)

    return lyr, ds
def split_by_lithology(label,input_path,out_put_path):
    data = gpd.read_file(input_path)
    all_lithology=data[label]
    lithology_list={}
    for i in all_lithology:
        lithology_list[i]=1
    lithology_list=list(lithology_list.keys())
    lithology_shp={}
    file_path_list=[]
    for lithology in lithology_list:
        lithology_shp[lithology]=data[data[label]==lithology]
    for name in lithology_list:
        temp=lithology_shp[name]
        path=out_put_path+"/{}.shp".format(str(name)).replace('|', '_')
        temp.to_file(path)
        file_path_list.append(str(path))
    return file_path_list
def vector_raster(input_layer, input_mask, output_layer):

    ds = gdal.Open(input_mask)
    geo_transform = ds.GetGeoTransform()

    cols = ds.RasterXSize
    rows = ds.RasterYSize

    lyr, ds1 = get_input_layer(input_layer)

    target_ds = gdal.GetDriverByName('GTiff').Create(output_layer, xsize=cols, ysize=rows, bands=1,
                                                     eType=gdal.GDT_Float32)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(ds.GetProjection())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
    del ds
    del target_ds

class Raster_INFO():
    def __init__(self,rst_rsc):
        rst_ds = gdal.Open(rst_rsc)
        self.geotrans = rst_ds.GetGeoTransform()
        self.xMin = self.geotrans[0]
        self.yMax = self.geotrans[3]
        self.pixelWidth = self.geotrans[1]
        self.pixelHeight = self.geotrans[5]
        self.rows = rst_ds.RasterYSize
        self.cols = rst_ds.RasterXSize
        self.xMax = self.xMin + (self.cols * self.pixelWidth)
        self.yMin = self.yMax + (self.rows * self.pixelHeight )
        band = rst_ds.GetRasterBand(1)
        self.nan = band.GetNoDataValue()
        #self.arr2d = band.ReadAsArray().astype("float")
        #self.msk = self.arr2d != self.nan
        #self.arr2d[~self.msk] = numpy.nan
        #rst_ds = None

def cal_rst_dist(vec_rst_rsc,dist_rst_dst):
    drv_tiff = gdal.GetDriverByName("GTiff") 
    vec_rst_info = Raster_INFO(vec_rst_rsc)
    dist_rst_ds = drv_tiff.Create(dist_rst_dst, 
                                 vec_rst_info.cols, vec_rst_info.rows,
                                 1, gdal.GDT_Float32)
    dist_rst_band = dist_rst_ds.GetRasterBand(1)
    dist_rst_ds.SetGeoTransform(vec_rst_info.geotrans)
    vec_rst_ds = gdal.Open(vec_rst_rsc)
    vec_rst_band = vec_rst_ds.GetRasterBand(1)
    gdal.ComputeProximity(vec_rst_band,dist_rst_band)
    dist_ras_ds = None
    vec_rst_ds = None

def pol2line(polyfn,linefn):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(polyfn,0)
    polyLayer = polyds.GetLayer()
    if os.path.exists(linefn):
        driver.DeleteDataSource(linefn)
    lineds =driver.CreateDataSource(linefn)
    linelayer = lineds.CreateLayer(linefn,polyLayer.GetSpatialRef(),geom_type = ogr.wkbLineString)
    featuredefn = linelayer.GetLayerDefn()
    for feat in polyLayer:
        geom = feat.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        outfeature = ogr.Feature(featuredefn)
        outfeature.SetGeometry(ring)
        linelayer.CreateFeature(outfeature)
        outfeature = None

if __name__ == '__main__':
    label='lithology'#岩性字段
    input_path='./Geologic.shp'
    out_put_path="./split"
    mask_path='./mask.tif'
    split_by_lithology(label,input_path,out_put_path)#按岩性分shp
    vector_raster('./split/Feldspathic quartzite _ minor siltite _ argillite.shp',mask_path,'Feldspathic quartzite _ minor siltite _ argillite_01.tif')#产生01tif，就是说在岩体内值1，外0。
    pol2line('./split/Feldspathic quartzite _ minor siltite _ argillite.shp','./Feldspathic quartzite _ minor siltite _ argillite_toline.shp')#面转线，岩体边缘是个线，方便计算内部外部距离。用01tif会导致岩体内部没有距离值
    vector_raster('./Feldspathic quartzite _ minor siltite _ argillite_toline.shp',mask_path,'./Feldspathic quartzite _ minor siltite _ argillite_toline.tif')
    cal_rst_dist('./Feldspathic quartzite _ minor siltite _ argillite_toline.tif','./Feldspathic quartzite _ minor siltite _ argillite_toline_dis.tif')#根据到岩体边缘的距离计算每个点权值。