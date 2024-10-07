import pandas as pd
from empath import Empath
import collections
import numpy as np
import os
from utils import *
import pathlib
from models import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader

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

def nn_k_fold(data: np.array, labels: np.array) -> tuple[np.array, np.array, np.array, np.array]:
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

    path = 'D:\School\ML\separated\Sentiment-Wellness-Tool\data\cnn_data\\'

    data_size = 0
    
    for count, key in enumerate(list(data_clustered.keys())):
        folder_path = path + key
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
        n_data = []
        data_size += 1

        for index, item in enumerate(data_clustered[key]):
            file_path = folder_path + f'\{data_size}.txt'
            file = open(file_path, 'w')
            file.close()


            if len(n_data) == 128:
                if data_size == (count + 1) * 1000:
                    save_list(file_path, n_data)
                    break

                save_list(file_path, n_data)
                n_data = []
                data_size += 1

            n_data.append(item)
        
            for i, item2 in enumerate(data_clustered[key]):
        
                if not i == index:
                    n_data.append(item2)
        
                if len(n_data) == 128:
                    if data_size == (count + 1) * 1000:
                        break

                    save_list(file_path, n_data)
                    file_path = folder_path + f'\{data_size}.txt'
                    n_data = []
                    data_size += 1
        
def get_set_labels(labels):
    set_labels = []
    for label in labels:
        if str(label) not in set_labels:
            set_labels.append(str(label))
    
    return set_labels

def status_splitter(input_file, path):
    data = pd.read_csv(input_file)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    # Step 2: Group by 'status' column
    grouped_data = data.groupby("status")

    # Step 3: Export each group to a separate CSV file
    for status, group in grouped_data:
        output_file = path + '\\' + f"{status}.csv"  # create a filename based on the status
        n_group = group.drop('Unnamed: 0', axis=1)
        n_group.to_csv(output_file, index=False)  # save the group to a CSV file

    return output_file

def data_creation(path_origin, path_destiny):
    for dirpath, _, files in os.walk(path_origin):
        for file in files:
            data = pd.read_csv(os.path.join(dirpath, file))
            f, labels = get_features(data)
            fdf = pd.DataFrame(f)
            combinations = generate_combinations(fdf, 128, 1000)
            folder = file.split('.')
            resulting_path = os.path.join(path_destiny, folder[0])
            pathlib.Path(resulting_path).mkdir(parents=True, exist_ok=True)
            for name, comb in enumerate(combinations):
                file_path = os.path.join(resulting_path, str(name) + '.txt')
                save_list(file_path, comb)
                
def cnn_k_fold(dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Create training and validation subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders for each subset
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64)

        # Initialize the model using the function
        model = cnn()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(10):  # Adjust number of epochs as needed
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs.unsqueeze(1))  # Add channel dimension
                optimizer.step()


            # Validation loop (optional)
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs_val, labels_val in val_loader:
                    outputs_val = model(inputs_val.unsqueeze(1))  # Add channel dimension
                    _, predicted = torch.max(outputs_val.data, 1)
                    total += labels_val.size(0)
                    correct += (predicted == labels_val).sum().item()

                print(f'Validation Accuracy: {100 * correct / total:.2f}%')


                     
class CustomDataset(Dataset):
    def __init__(self, base_dir):
        self.data = []
        self.labels = []
        self.label_map = {}
        
        # Load data from folders
        for label_idx, label in enumerate(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, label)
            if os.path.isdir(folder_path):
                self.label_map[label_idx] = label  # Map index to class name
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):  # Adjust based on your file format
                        file_path = os.path.join(folder_path, filename)
                        data = []
                        with open(file_path, 'r') as f:
                            for line in f.readlines():
                                can, dline = load_list(line)
                                if can:
                                    data.append(dline)
                                else:
                                    raise TypeError(f'Error parsing in position: {len(dline)}.')
                        data_entry = np.array(data).astype(np.float32)
                        self.data.append(data_entry)
                        self.labels.append(label_idx)  # Use index as the label

        # Convert to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




# Example of iterating through the DataLoader

