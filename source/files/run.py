# start clustering

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
 

# make matplotlib plot inline (Only in Ipython).

path_transformed_df = 'data\kmeans_data\data.csv'
path_reduced_df = 'data\kmeans_data\data_reduced.csv'
path_labels = 'data\labels.txt'


raw_data = pd.read_csv("data\\raw_data\data.csv")
path_status_splitted = 'data\status_splitted'
path_original_augmented_data = 'data\\augmented_data'

# labels = raw_data['status']

labels = []

# save_lazy(path_labels, labels)

with open(path_labels, 'r') as file:
    for line in file.readlines():
       for word in line.split(','):
           labels.append(word)

# print(labels)

# print('finish')

# features, labels = get_features(raw_data)

# status_splitter("data\\raw_data\data.csv", path_status_splitted)
# print('creating splitted datasets')

# dataframe = pd.DataFrame(features)
# dataframe.to_csv(path_transformed_df, index=False)

# dataframe = pd.read_csv(path_reduced_df)

# data_creation(path_status_splitted, 'data\splitted_data')

base_directory = 'data\splitted_data'  # Change this to your actual path
dataset = CustomDataset(base_directory)


cnn_k_fold(dataset)
# print('test')

# correlation = dataframe.corr()

# threshold = 0.8
# filtered_correlation = correlation[(correlation >= threshold) | (correlation <= -threshold)]

# plt.figure(figsize=(12, 10))  # Set the figure size to accommodate many features
# sns.heatmap(filtered_correlation, annot=False, cmap='coolwarm', fmt=".2f", square=True)
# plt.title('Correlation Matrix Heatmap')
# plt.xlabel('Features')
# plt.ylabel('Features')
# plt.show()

# to_drop = set()  # Set to hold features to drop

# for i in range(len(correlation.columns)):
#     for j in range(i):
#         if correlation.iloc[i, j] > threshold:
#             colname = correlation.columns[i]  # Get column name
#             to_drop.add(colname)  # Add to drop set

# dataframe_reduced = dataframe.drop(columns=to_drop)


# dataframe_reduced.to_csv(path_reduced_df, index=False)

# print(dataframe)
# n_features = transform_features(features)
# n_labels = labels_transform(labels)


# print('testing data')

# dataframe.reset_index(drop=True, inplace=True)

# warnings.filterwarnings('ignore')

# X = dataframe.values
# print(X)

# Y = dataframe.columns.values

# km_model = kmeans()

# km_model.fit(X)

# labels_ = [labels[k] for k in km_model.labels_]

# tags = []
# for label in labels_:
#     if label not in tags:
#         tags.append(label)

# print(tags)

# nnormal = [n for n in labels_ if n == 'Normal']
# ndepre = [n for n in labels_ if n == 'Depression']

# path_number = 'D:\School\ML\separated\Sentiment-Wellness-Tool\data\\nn_data\\numbers.txt'
# with open(path_number, 'w') as file:
#     to_write = f'{len(nnormal)} {len(ndepre)}'
#     file.write(to_write)

# print('Check rare data')

# print(labels_)

# data_augmentation(X, labels_)

# Grafica la distribución de datos (_ground truth_) usando matplotlib `scatter(axis-x, axis-y, color)`.
# plt.scatter(X[:,0], X[:,1])

# plt.scatter(X[:,0], X[:,1], c=km_model.labels_)
# plt.scatter(km_model.cluster_centers_[:,0], 
#             km_model.cluster_centers_[:,1], 
#             c='w', marker='x', linewidths=2)

# plt.show()

# print('test_kmeans')