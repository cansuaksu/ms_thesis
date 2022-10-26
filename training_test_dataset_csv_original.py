import rasterio.mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import fiona
import itertools
import pandas as pd
import numpy as np
import glob
import os
'''THIS SCRIPT IS FOR THE INPUT COMBINATIONS OF ATMO_TOPO AND SC_ONLY. SIMILAR '''

'''BEFORE THIS SCRIPT, GETTING A COMPOSITE IMAGE VIA ARCMAP IS NEEDED'''


path = "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/23_Jan_2020_sc_only/"

def get_test_data(path):
    with fiona.open(path + "/Test_Region/Clip_Frame.shp", "r") as shapefile:
        for feature in shapefile:
            shapes = [feature["geometry"]]

    with rasterio.open(path + "/S2_Bands_TIFF/Composite.tif") as src:
        out_image, out_trasform = rasterio.mask.mask(src, shapes, crop="True")
        out_meta = src.meta

    out_meta.update({
        'driver': 'Gtiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_trasform
    })

    with rasterio.open(path + "/Test_Region/Composite_Clipped.tif", "w", **out_meta) as dst:
        dst.write(out_image)
    array = rasterio.open(path + "Test_Region/Composite_Clipped.tif").read()
    shape_list = list(array.shape)
    array = array.reshape(shape_list[0], shape_list[1] * shape_list[2])
    array = array.transpose()
    df = pd.DataFrame(array,columns=['band2', 'band3', 'band4', 'band5', 'band6', 'band7', 'band8a', 'band11', 'band12','ndsi', 'ndvi', 'ndwi'])
    df.to_csv(path + "Test_Region/Test_Data_original.csv", index=False)

#
# get_test_data(path)


def training_polygons(path, choice="Cloud"):
    with fiona.open(path+"/Training_Polygons/" + choice + ".shp") as input:
        # preserve the schema of the original shapefile, including the crs
        meta = input.meta
        with fiona.open(path+ "/Training_Polygons/" + choice + "_dissolve.shp" ,'w', **meta) as output:
            # groupby clusters consecutive elements of an iterable which have the same key so you must first sort the features by the 'STATEFP' field
            e = sorted(input, key=lambda k: k['properties']['Id'])
            # group by the 'STATEFP' field
            for key, group in itertools.groupby(e, key=lambda x: x['properties']['Id']):
                properties, geom = zip(*[(feature['properties'], shape(feature['geometry'])) for feature in group])
                # write the feature, computing the unary_union of the elements in the group with the properties of the first element in the group
                output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})

    with fiona.open(path+ "/Training_Polygons/" + choice + "_dissolve.shp", "r") as shapefile:
        for feature in shapefile:
            shapes = [feature["geometry"]]

    with rasterio.open(path + "/S2_Bands_TIFF/Composite.tif") as src:
        out_image, out_trasform = rasterio.mask.mask(src, shapes, crop="True")
        out_meta = src.meta

    out_meta.update({
        'driver': 'Gtiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_trasform
    })

    with rasterio.open(path +"Training_Polygons/python/Composite_Polygons/" + choice + ".tif", "w", **out_meta) as dst:
        dst.write(out_image)

# training_polygons(path, choice="Water")

arr = rasterio.open(path + "/Training_Polygons/python/Composite_Polygons/Water.tif").read()
# print(arr)
nodata=arr[0][0][0] ###For obtaining nodata.
print(nodata)

def training_polygon_to_csv(path, choice="Cloud", choice2=1):
    array= rasterio.open(path + "/Training_Polygons/python/Composite_Polygons/" + choice + ".tif").read()
    shape_list = list(array.shape)
    array = array.reshape(shape_list[0], shape_list[1]*shape_list[2])
    array = array.transpose()
    df = pd.DataFrame(array, columns = ['band2','band3','band4', 'band5', 'band6', 'band7', 'band8a', 'band11', 'band12','ndsi', 'ndvi', 'ndwi'])
    df.drop(df.loc[df['band2'] == nodata].index, inplace=True)
    df.insert(0, "class", choice2)
    df.to_csv(path + "/Training_Polygons/python/Composite_Polygons/" + choice + ".csv", index=False)
#
# training_polygon_to_csv(path, choice="Water", choice2=4)

def sample_csv(path, choice="Cloud", sample_num=1000):
    total_training = pd.read_csv(path + "/Training_Polygons/python/Composite_Polygons/" + choice + ".csv")

    total_training= total_training.sample(sample_num)
    total_training.to_csv(path + "/Training_Polygons/Original/" + choice + "_" + str(sample_num) + ".csv", index=False)

# sample_csv(path, choice="Water", sample_num=1000)

def concat_csvs(path, sample_num=1000):
    files = os.path.join(path + "/Training_Polygons/Original/" , "*" + str(sample_num) + ".csv")

    # list of merged files returned
    files = glob.glob(files)


    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(path + "/Training_Polygons/Original/Training_Data" + "_" + str(sample_num) + ".csv", index=False)
# #
# concat_csvs(path, sample_num=1000)

















'''ORIGINAL CODE IS BELOW FOR TRAINING POLYGONS'''
# with fiona.open('D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/01_ALPS/Work_Folder/5_Dec_2018_sc_only/Training_Polygons/Cloud.shp') as input:
#     # preserve the schema of the original shapefile, including the crs
#     meta = input.meta
#     with fiona.open('D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/01_ALPS/Work_Folder/5_Dec_2018_sc_only/Training_Polygons/Cloud_dissolve.shp', 'w', **meta) as output:
#         # groupby clusters consecutive elements of an iterable which have the same key so you must first sort the features by the 'STATEFP' field
#         e = sorted(input, key=lambda k: k['properties']['Id'])
#         # group by the 'STATEFP' field
#         for key, group in itertools.groupby(e, key=lambda x:x['properties']['Id']):
#             properties, geom = zip(*[(feature['properties'],shape(feature['geometry'])) for feature in group])
#             # write the feature, computing the unary_union of the elements in the group with the properties of the first element in the group
#             output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})
#
# with fiona.open("D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/01_ALPS/Work_Folder/5_Dec_2018_sc_only/Training_Polygons/Cloud_dissolve.shp", "r") as shapefile:
#     for feature in shapefile:
#         shapes = [feature["geometry"]]
#
#
#
# with rasterio.open("D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/01_ALPS/Work_Folder/5_Dec_2018_atmo_topo/S2_Bands_TIFF/Composite.tif") as src:
#     out_image, out_trasform = rasterio.mask.mask(src, shapes, crop="True")
#     out_meta = src.meta
#
# out_meta.update({
#     'driver': 'Gtiff',
#     'height':out_image.shape[1],
#     'width': out_image.shape[2],
#     'transform': out_trasform
# })
#
# with rasterio.open('D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/01_ALPS/Work_Folder/5_Dec_2018_atmo_topo/Training_Polygons/python/Composite_Polygons/Cloud.tif', "w", **out_meta) as dst:
#     dst.write(out_image)


'''IF YOU NEED TO CHECK ANYTHING, CHECK BELOW'''
# shape_list = list(arr.shape)
# arr = arr.reshape(shape_list[0], shape_list[1]*shape_list[2])
# arr = arr.transpose()
# print(arr.shape)
#
# df = pd.DataFrame(arr, columns = ['band2','band3','band4', 'band5', 'band6', 'band7', 'band8a', 'band11', 'band12','ndsi', 'ndvi', 'ndwi', 'dem'])
#
# nodata = df.loc[0]["band2"]
# df.drop(df.loc[df['band2']==nodata].index, inplace=True)
#
# print(df)

# df.to_csv(path + "/Training_Polygons/python/Composite_Polygons/Cloud.csv", index=False)


