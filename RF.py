
import rasterio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

'''THIS CODE IS FOR ATMO_TOPO AND SC_ONLY INPUT COMBINATIONS ONLY'''
start = time.time()

img_path= "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_sc_only/Test_Area/"
train_path= "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_sc_only/Training_Polygons/Original/Training_Data_1000.csv"
save_path= "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_sc_only/220610_Classified_01/"
# img = rasterio.open(img_path).read()
# print(img.shape)
def random_forest_og(img_path, train_path, save_path, sample_size = 300, choice="atmo_topo"):
    img = rasterio.open(img_path)
    img_as_array = img.read()
    input_crs = img.crs
    input_gt = img.transform
    df_train = pd.read_csv(train_path)

    # Initialize our model with 500 trees
    X = df_train.drop('class', axis = 1)
    y = df_train['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Fit our model to training data
    rf = RandomForestClassifier(n_estimators=500, oob_score=True)
    rf_model = rf.fit(X_train, y_train)
    arr1 = img_as_array.reshape(12,1001*1001).transpose()
    class_prediction = rf_model.predict(arr1)
    class_prediction = class_prediction.reshape(1001,1001)
    class_prediction = class_prediction.astype(np.float32)
    with rasterio.open(
        save_path + '/RF_classification_'+ choice +'_'+ str(sample_size) +'runtime.tif',
        'w',
        driver='GTiff',
        height=img.shape[0],
        width=img.shape[1],
        count=1,
        dtype=np.float32,
        crs=input_crs,
        transform=input_gt,
    ) as dest_file:
        dest_file.write(class_prediction, 1)
    dest_file.close()
random_forest_og(img_path, train_path, save_path,sample_size=1000, choice='atmo_topo')
end = time.time()
print((end - start)/60)

# print(save_path + '/RF_classification_'+ "atmo_topo" +'_'+ str(1000) +'new.tif')
