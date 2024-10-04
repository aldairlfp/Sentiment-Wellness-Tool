import pandas as pd
from empath import Empath 
import collections 
import numpy as np
from utils import save_list
import os

from sklearn.model_selection import StratifiedKFold

def get_features(data):
    data_mod = data.dropna()

    lexicon = Empath()
    result = []
    labels = []

    for data in data_mod['statement']:
        result.append(lexicon.analyze(data))
    for label in data_mod['status']:
        labels.append(label)

    return result, labels

def transform_features(features: list[dict]) -> np.array:
    data = []
    for item in features:
        temp = []

        for k in item:
            temp.append(item[k])

        data.append(temp)
    data = np.array(data)

    return data        

def labels_transform(labels: list) -> np.array:
    classes = []
    idx_to_labels = {}
    labels_to_idx = {}
    p_labels = []

    for label in labels:
        i = 0
    
        if label not in classes:
            classes.append(label)
            idx_to_labels[i] = label
            labels_to_idx[label] = i

        p_labels.append(labels_to_idx[label])
    
        i += 1

    p_labels = np.array(p_labels)

    return p_labels

def k_fold(data: np.array, labels: np.array) -> tuple[np.array, np.array, np.array, np.array]:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

    for fold, (train_index, val_index) in enumerate(skf.split(data, labels)):
        print(f"Fold {fold + 1}/{3}")

        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        return X_train, y_train, X_val, y_val
    


def data_augmentation(data, labels):
    data_clustered = {}

    for index, label in enumerate(list(labels)):
        if label in list(data_clustered.keys()):
            data_clustered[label].append(data[index])
        else:
            data_clustered[label] = [data[index]]

    path = '.\\data\\kmeans_data\\'

    data_size = 0
    
    for count, key in enumerate(list(data_clustered.keys())):
        folder_path = path + key
        os.mkdir(path = folder_path)
        n_data = []
        for index, item in enumerate(data_clustered[key]):
            file_path = folder_path + f'{index}.txt'
            file = open(file_path, 'w')
            file.close()

            if len(n_data) == 128:
                save_list(file_path, n_data)
                n_data = []
                data_size += 1
                if data_size == (count + 1) * 1000:
                    break

            n_data.append(item)
        
            for i, item2 in enumerate(data_clustered[key]):
        
                if not i == index:
                    n_data.append(item2)
        
                if len(n_data) == 128:
                    save_list(file_path, n_data)
                    n_data = []
                    data_size += 1
                    if data_size == (count + 1) * 1000:
                        break
        
