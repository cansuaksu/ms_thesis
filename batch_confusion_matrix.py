import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


path = "D:/Drivers/GGIT/SK_TEZ_110622/Cansu_Tez_Draft/02_TATRA/Work_Folder/8_Apr_2019_atmo_topo"

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def conf_matrix_batch(path=path):
    path_read = path + "/Accuracy_Assessment/ALL/"
    # df_atmo_topo_1000 = pd.read_csv(path_read + "classified_atmo_topo_1000.csv")
    # df_atmo_topo_300=pd.read_csv(path_read + "classified_atmo_topo_300.csv")
    # df_sc_1000=pd.read_csv(path_read + "classified_sc_1000.csv")
    # df_sc_300 = pd.read_csv(path_read + "classified_sc_300.csv")
    # df_dem_1000 = pd.read_csv(path_read + "classified_dem_1000.csv")
    # df_dem_300 = pd.read_csv(path_read + "classified_dem_300.csv")
    # df_pca_1000 = pd.read_csv(path_read + "classified_pca_1000.csv")
    # df_pca_300 = pd.read_csv(path_read + "classified_pca_300.csv")
    df_pca_plus_1000 = pd.read_csv(path_read + "classified_pca_plus_1000.csv")
    df_pca_plus_300 = pd.read_csv(path_read + "classified_pca_plus_300.csv")
    df_pca_plus2_1000 = pd.read_csv(path_read + "classified_pca_plus_10002.csv")
    df_pca_plus2_300 = pd.read_csv(path_read + "classified_pca_plus_3002.csv")

    df_gt = pd.read_csv(path_read + "groundtruth.csv")

    # cf_matrix_atmo_topo_1000 = confusion_matrix(df_gt, df_atmo_topo_1000)
    # cf_matrix_atmo_topo_300 = confusion_matrix(df_gt, df_atmo_topo_300)
    # cf_matrix_sc_1000 = confusion_matrix(df_gt, df_sc_1000)
    # cf_matrix_sc_300 = confusion_matrix(df_gt, df_sc_300)
    # cf_matrix_dem_1000 = confusion_matrix(df_gt, df_dem_1000)
    # cf_matrix_dem_300 = confusion_matrix(df_gt, df_dem_300)
    # cf_matrix_pca_1000 = confusion_matrix(df_gt, df_pca_1000)
    # cf_matrix_pca_300 = confusion_matrix(df_gt, df_pca_300)
    cf_matrix_pca_plus_1000 = confusion_matrix(df_gt, df_pca_plus_1000)
    cf_matrix_pca_plus_300 = confusion_matrix(df_gt, df_pca_plus_300)
    cf_matrix_pca_plus2_1000 = confusion_matrix(df_gt, df_pca_plus2_1000)
    cf_matrix_pca_plus2_300 = confusion_matrix(df_gt, df_pca_plus2_300)

    # cm_df_atmo_topo_1000 = pd.DataFrame(cf_matrix_atmo_topo_1000,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    #
    # cm_df_atmo_topo_300 = pd.DataFrame(cf_matrix_atmo_topo_300,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    #
    # cm_df_sc_1000 = pd.DataFrame(cf_matrix_sc_1000,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    #
    # cm_df_sc_300= pd.DataFrame(cf_matrix_sc_300,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    # cm_df_dem_1000 = pd.DataFrame(cf_matrix_dem_1000,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    # cm_df_dem_300 = pd.DataFrame(cf_matrix_dem_300,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    #
    # cm_df_pca_1000 = pd.DataFrame(cf_matrix_pca_1000,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])
    # cm_df_pca_300 = pd.DataFrame(cf_matrix_pca_300,
    #                                     index=["cloud", "land", "snow", "water"],
    #                                     columns=["cloud", "land", "snow", "water"])

    cm_df_pca_plus_1000 = pd.DataFrame(cf_matrix_pca_plus_1000,
                                  index=["cloud", "land", "snow", "water"],
                                  columns=["cloud", "land", "snow", "water"])
    cm_df_pca_plus_300 = pd.DataFrame(cf_matrix_pca_plus_300,
                                 index=["cloud", "land", "snow", "water"],
                                 columns=["cloud", "land", "snow", "water"])
    cm_df_pca_plus2_1000 = pd.DataFrame(cf_matrix_pca_plus2_1000,
                                  index=["cloud", "land", "snow", "water"],
                                  columns=["cloud", "land", "snow", "water"])
    cm_df_pca_plus2_300 = pd.DataFrame(cf_matrix_pca_plus2_300,
                                 index=["cloud", "land", "snow", "water"],
                                 columns=["cloud", "land", "snow", "water"])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize= (12,12))

    # sns.heatmap(cm_df_atmo_topo_1000, annot=True, fmt='g', ax=axes[0,0]).set(title="Atmo_Topo_1000")
    # sns.heatmap(cm_df_sc_1000, annot=True, fmt='g', ax=axes[0,1]).set(title="SC_1000")
    # sns.heatmap(cm_df_dem_1000, annot=True, fmt='g', ax=axes[0,2]).set(title="DEM_1000")
    # sns.heatmap(cm_df_pca_1000, annot=True, fmt='g', ax=axes[0,3]).set(title="PCA_1000")
    sns.heatmap(cm_df_pca_plus_1000, annot=True, fmt='g', ax=axes[0, 0]).set(title="PCA_PLUS_1000")
    sns.heatmap(cm_df_pca_plus2_1000, annot=True, fmt='g', ax=axes[0, 1]).set(title="PCA_PLUS2_1000")

    # sns.heatmap(cm_df_atmo_topo_300, annot=True, fmt='g', ax=axes[1,0]).set(title="Atmo_Topo_300")
    # sns.heatmap(cm_df_sc_300, annot=True, fmt='g', ax=axes[1,1]).set(title="SC_300")
    # sns.heatmap(cm_df_dem_300, annot=True, fmt='g', ax=axes[1,2]).set(title="DEM_300")
    # sns.heatmap(cm_df_pca_300, annot=True, fmt='g', ax=axes[1,3]).set(title="PCA_300")
    sns.heatmap(cm_df_pca_plus_300, annot=True, fmt='g', ax=axes[1, 0]).set(title="PCA_PLUS_300")
    sns.heatmap(cm_df_pca_plus2_300, annot=True, fmt='g', ax=axes[1, 1]).set(title="PCA_PLUS2_300")

    fig.suptitle("TATRA - 8 April 2019")
    fig.tight_layout()
    plt.show()



def acc_kappa(path):
    path_read = path + "/Accuracy_Assessment/ALL/"
    # df_atmo_topo_1000 = pd.read_csv(path_read + "classified_atmo_topo_1000.csv")
    # df_atmo_topo_300 = pd.read_csv(path_read + "classified_atmo_topo_300.csv")
    # df_sc_1000 = pd.read_csv(path_read + "classified_sc_1000.csv")
    # df_sc_300 = pd.read_csv(path_read + "classified_sc_300.csv")
    # df_dem_1000 = pd.read_csv(path_read + "classified_dem_1000.csv")
    # df_dem_300 = pd.read_csv(path_read + "classified_dem_300.csv")
    # df_pca_1000 = pd.read_csv(path_read + "classified_pca_1000.csv")
    # df_pca_300 = pd.read_csv(path_read + "classified_pca_300.csv")
    df_pca_plus_1000 = pd.read_csv(path_read + "classified_pca_plus_1000.csv")
    df_pca_plus_300 = pd.read_csv(path_read + "classified_pca_plus_300.csv")
    df_pca_plus2_1000 = pd.read_csv(path_read + "classified_pca_plus_10002.csv")
    df_pca_plus2_300 = pd.read_csv(path_read + "classified_pca_plus_3002.csv")
    df_gt = pd.read_csv(path_read + "groundtruth.csv")

    # overall_acc_atmo_topo_1000 = accuracy_score(df_gt, df_atmo_topo_1000)
    # kappa_atmo_topo_1000 = cohen_kappa_score(df_gt, df_atmo_topo_1000)
    #
    # overall_acc_atmo_topo_300 = accuracy_score(df_gt, df_atmo_topo_300)
    # kappa_atmo_topo_300 = cohen_kappa_score(df_gt, df_atmo_topo_300)
    #
    # overall_acc_sc_1000 = accuracy_score(df_gt, df_sc_1000)
    # kappa_sc_1000 = cohen_kappa_score(df_gt, df_sc_1000)
    #
    # overall_acc_sc_300 = accuracy_score(df_gt, df_sc_300)
    # kappa_sc_300 = cohen_kappa_score(df_gt, df_sc_300)
    #
    # overall_acc_dem_1000 = accuracy_score(df_gt, df_dem_1000)
    # kappa_dem_1000 = cohen_kappa_score(df_gt, df_dem_1000)
    #
    # overall_acc_dem_300 = accuracy_score(df_gt, df_dem_300)
    # kappa_dem_300 = cohen_kappa_score(df_gt, df_dem_300)
    #
    # overall_acc_pca_1000 = accuracy_score(df_gt, df_pca_1000)
    # kappa_pca_1000 = cohen_kappa_score(df_gt, df_pca_1000)
    #
    # overall_acc_pca_300 = accuracy_score(df_gt, df_pca_300)
    # kappa_pca_300 = cohen_kappa_score(df_gt, df_pca_300)

    overall_acc_pca_plus_1000 = accuracy_score(df_gt, df_pca_plus_1000)
    kappa_pca_plus_1000 = cohen_kappa_score(df_gt, df_pca_plus_1000)

    overall_acc_pca_plus_300 = accuracy_score(df_gt, df_pca_plus_300)
    kappa_pca_plus_300 = cohen_kappa_score(df_gt, df_pca_plus_300)

    overall_acc_pca_plus2_1000 = accuracy_score(df_gt, df_pca_plus2_1000)
    kappa_pca_plus2_1000 = cohen_kappa_score(df_gt, df_pca_plus2_1000)

    overall_acc_pca_plus2_300 = accuracy_score(df_gt, df_pca_plus2_300)
    kappa_pca_plus2_300 = cohen_kappa_score(df_gt, df_pca_plus2_300)

    # Numbers of pairs of bars you want
    N = 4
    # Y values
    list_overall_acc = [overall_acc_pca_plus_1000, overall_acc_pca_plus_300, overall_acc_pca_plus2_1000, overall_acc_pca_plus2_300]
    list_kappa = [kappa_pca_plus_1000, kappa_pca_plus_300, kappa_pca_plus2_1000, kappa_pca_plus2_300]
    # list_overall_acc = [overall_acc_atmo_topo_1000, overall_acc_atmo_topo_300, overall_acc_sc_1000, overall_acc_sc_300,
    #                     overall_acc_dem_1000, overall_acc_dem_300, overall_acc_pca_1000, overall_acc_pca_300]
    # list_kappa = [kappa_atmo_topo_1000, kappa_atmo_topo_300, kappa_sc_1000, kappa_sc_300, kappa_dem_1000, kappa_dem_300,
    #               kappa_pca_1000, kappa_pca_300]

    # X values

    # x_values = ['Atmo_Topo_1000', 'Atmo_Topo_300', 'SC_1000', 'SC_300', 'DEM_1000', 'DEM_300', 'PCA_1000', 'PCA_300']
    x_values = ['PCA_PLUS_1000', 'PCA_PLUS_300', 'PCA_PLUS2_1000', 'PCA_PLUS2_300']

    # Position of bars on x-axis
    ind = np.arange(N)
    plt.figure(figsize=(20, 10))
    width = 0.22

    plt.bar(ind, list_overall_acc, width, label='Overall Accuracy', color="#FE5E03")
    plt.bar(ind + width, list_kappa, width, label='Kappa Coefficient', color="#03FEE1")

    plt.xlabel('Methodology')
    plt.ylabel('Overall Accuracy and Kappa Coefficient Values')
    plt.title("TATRA - 8 April 2019")
    plt.xticks(ind + width / 2, x_values)
    plt.ylim(min(list_kappa) - 0.1, max(list_overall_acc) + 0.1)
    # plt.yticks(np.arange(min(list_kappa)+0.1,max(list_overall_acc), 0.01))

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.show()


def to_csv(path):
    path_read = path + "/Accuracy_Assessment/ALL/"
    df_atmo_topo_1000 = pd.read_csv(path_read + "classified_atmo_topo_1000.csv")
    df_atmo_topo_300 = pd.read_csv(path_read + "classified_atmo_topo_300.csv")
    df_sc_1000 = pd.read_csv(path_read + "classified_sc_1000.csv")
    df_sc_300 = pd.read_csv(path_read + "classified_sc_300.csv")
    df_dem_1000 = pd.read_csv(path_read + "classified_dem_1000.csv")
    df_dem_300 = pd.read_csv(path_read + "classified_dem_300.csv")
    df_pca_1000 = pd.read_csv(path_read + "classified_pca_1000.csv")
    df_pca_300 = pd.read_csv(path_read + "classified_pca_300.csv")
    # df_pca_plus_1000 = pd.read_csv(path_read + "classified_pca_plus_1000.csv")
    # df_pca_plus_300 = pd.read_csv(path_read + "classified_pca_plus_300.csv")
    # df_pca_plus2_1000 = pd.read_csv(path_read + "classified_pca_plus_10002.csv")
    # df_pca_plus2_300 = pd.read_csv(path_read + "classified_pca_plus_3002.csv")

    df_gt = pd.read_csv(path_read + "groundtruth.csv")

    overall_acc_atmo_topo_1000 = accuracy_score(df_gt, df_atmo_topo_1000)
    kappa_atmo_topo_1000 = cohen_kappa_score(df_gt, df_atmo_topo_1000)

    overall_acc_atmo_topo_300 = accuracy_score(df_gt, df_atmo_topo_300)
    kappa_atmo_topo_300 = cohen_kappa_score(df_gt, df_atmo_topo_300)

    overall_acc_sc_1000 = accuracy_score(df_gt, df_sc_1000)
    kappa_sc_1000 = cohen_kappa_score(df_gt, df_sc_1000)

    overall_acc_sc_300 = accuracy_score(df_gt, df_sc_300)
    kappa_sc_300 = cohen_kappa_score(df_gt, df_sc_300)

    overall_acc_dem_1000 = accuracy_score(df_gt, df_dem_1000)
    kappa_dem_1000 = cohen_kappa_score(df_gt, df_dem_1000)

    overall_acc_dem_300 = accuracy_score(df_gt, df_dem_300)
    kappa_dem_300 = cohen_kappa_score(df_gt, df_dem_300)

    overall_acc_pca_1000 = accuracy_score(df_gt, df_pca_1000)
    kappa_pca_1000 = cohen_kappa_score(df_gt, df_pca_1000)

    overall_acc_pca_300 = accuracy_score(df_gt, df_pca_300)
    kappa_pca_300 = cohen_kappa_score(df_gt, df_pca_300)

    # overall_acc_pca_plus_1000 = accuracy_score(df_gt, df_pca_plus_1000)
    # kappa_pca_plus_1000 = cohen_kappa_score(df_gt, df_pca_plus_1000)
    #
    # overall_acc_pca_plus_300 = accuracy_score(df_gt, df_pca_plus_300)
    # kappa_pca_plus_300 = cohen_kappa_score(df_gt, df_pca_plus_300)
    #
    # overall_acc_pca_plus2_1000 = accuracy_score(df_gt, df_pca_plus2_1000)
    # kappa_pca_plus2_1000 = cohen_kappa_score(df_gt, df_pca_plus2_1000)
    #
    # overall_acc_pca_plus2_300 = accuracy_score(df_gt, df_pca_plus2_300)
    # kappa_pca_plus2_300 = cohen_kappa_score(df_gt, df_pca_plus2_300)

    dict_ult = {'Name': ['Overall Accuracy', 'Kappa'], 'Atmo_Topo_1000': [overall_acc_atmo_topo_1000, kappa_atmo_topo_1000],'Atmo_Topo_300': [overall_acc_atmo_topo_300, kappa_atmo_topo_300], 'SC_1000': [overall_acc_sc_1000, kappa_sc_1000], 'SC_300':[overall_acc_sc_300, kappa_sc_300], 'DEM_1000': [overall_acc_dem_1000,kappa_dem_1000], 'DEM_300': [overall_acc_dem_300, kappa_dem_300], 'PCA_1000': [overall_acc_pca_1000, kappa_pca_1000], 'PCA_300': [overall_acc_pca_300,kappa_pca_300] }

    df = pd.DataFrame(dict_ult)
    df.to_csv("E:/TEZ/Presentations/140722/Alps_24_Jan_2019_jun.csv")

conf_matrix_batch(path)
# acc_kappa(path)
# to_csv(path)

