# msthesis

# This repository is for the Python scripts that I have used for my thesis, titled: 
# "ASSESSMENT OF RANDOM FOREST METHOD IN PIXEL-BASED SNOW COVER CLASSIFICATION IN ALPINE REGION, TATRA MOUNTAINS AND KAÇKAR MOUNTAINS"
# This study researchs the accuracy of Random Forest algorithm for Sentinel-2 images, employing different input combinations.
# These are some of the main scripts I have used in order to investigate the accuracy of Random Forest on remote sensing images (Sentinel-2 images):
# training_test_dataset_csv.py  - Preparation of training and test datasets
# RF.py - For the Random Forest algorithm (used on Sentinel-2 images, returns a classified raster (.tiff) image in return
# pca_code.py - For obtaining principal components from a composite Sentinel-2 image (9 Sentinel-2 bands compiled as one raster image in a different GIS platform 
# before using this code)
# batch_confusion_matrix - This script is for:
#   1- Obtaining confusion matrices from classified images and displaying/saving multiple confusion matrices on a singe image file
#   2- Obtaining Overall Accuracy and Kappa Coefficient values, and displaying them on a graph, side by side, for each of 9 images for different input combinations
#   3- Saving Overall Accuracy and Kappa Coefficient values to individual .csv files

