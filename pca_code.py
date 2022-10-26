from sklearn.decomposition import PCA
import skimage
import os
import rasterio
import numpy as np
import time

''' COMPOSITE RASTER SHOULD BE OBTAINED BEFOREHAND'''

start = time.time()

path_composite = "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_atmo_topo/S2_Bands_PCA/Comp_PCA.tif"
save_path = "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_atmo_topo/S2_Bands_PCA/"

meta_data = rasterio.open(path_composite)
comp_array = rasterio.open(path_composite).read()
# print(comp_array.shape) # 6, 5490, 5490
comp=np.reshape(comp_array,(6,5490*5490))

comp_final = np.transpose(comp)
# comp_final[np.isnan(comp_final)] = 32767
pca = PCA(n_components = 3)

a=pca.fit_transform(comp_final)
pca1_np = a[:,0]
pca1 = np.reshape(pca1_np, (5490,5490))
kwargs = meta_data.meta

with rasterio.open(os.path.join(save_path, 'PCA1_runtime.tif'), 'w', **kwargs) as dst:
    dst.write_band(1, pca1)

pca2_np = a[:,1]
pca2 = np.reshape(pca2_np, (5490,5490))

with rasterio.open(os.path.join(save_path, 'PCA2_runtime.tif'), 'w', **kwargs) as dst:
    dst.write_band(1, pca2)

pca3_np = a[:,2]
pca3 = np.reshape(pca3_np, (5490,5490))

with rasterio.open(os.path.join(save_path, 'PCA3_runtime.tif'), 'w', **kwargs) as dst:
    dst.write_band(1, pca3)

end = time.time()
print((end - start)/60)

# pca = PCA(n_components=3)
# pca.fit(comp_array)
# predict = pca.transform(comp_array)


