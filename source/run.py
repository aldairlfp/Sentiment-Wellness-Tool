# Start from importing necessary packages.
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from IPython.display import display
from sklearn import metrics # for evaluations
from sklearn.datasets import make_blobs, make_circles # for generating experimental data
from sklearn.preprocessing import StandardScaler # for feature scaling
import pandas as pd

from data_processing import *
from utils import *
from models import *
 

path_transformed_df = '/media/jose/A63C16883C1654211/Proyectos/Sentiment-Wellness-Tool/data/kmeans_data/data.csv' 
# print("j")
# raw_data = pd.read_csv('/media/jose/A63C16883C1654211/Proyectos/Sentiment-Wellness-Tool/data/raw_data/data.csv')
# print("pp")
# features, labels = get_features(raw_data)


# dataframe = pd.DataFrame(features)

# dataframe.to_csv(path_transformed_df + '/data.csv')

# #dataframe = pd.read_csv(path_transformed_df)

# HASTA ACA LOS DATOS ORIGINALES QUEDAN CON LAS DIMENSIONES REDUCIDAS A PARTIR DEL DICCIONARIO PSIC

# dataframe = pd.read_csv(path_transformed_df)
# correlation = dataframe.corr()

# threshold = 0.8
# filtered_correlation = correlation[(correlation >= threshold) | (correlation <= -threshold)]
# plt.figure(figsize=(12, 10))  # Set the figure size to accommodate many features
# sns.heatmap(filtered_correlation, annot=False, cmap='coolwarm', fmt=".2f", square=True)
# plt.title('Correlation Matrix Heatmap')
# plt.xlabel('Features')
# plt.ylabel('Features')
# plt.show()

# print('visualize')

# path_reduced_df = '/media/jose/A63C16883C1654211/Proyectos/Sentiment-Wellness-Tool/data/kmeans_data/reduced_data.csv'
# to_drop = set()  # Set to hold features to drop

# for i in range(len(correlation.columns)):
#     for j in range(i):
#         if correlation.iloc[i, j] > threshold:
#             colname = correlation.columns[i]  # Get column name
#             to_drop.add(colname)  # Add to drop set
# print("aaaaaaaaa")
# dataframe_reduced = dataframe.drop(columns=to_drop)

# dataframe_reduced.to_csv(path_reduced_df)
# print("skdnskd")
# # HASTA ACA SEGUIMOS REDUCIENDO DIMENSIONES SEGUN LA CORRELACION DE LAS ETIQUETAS



