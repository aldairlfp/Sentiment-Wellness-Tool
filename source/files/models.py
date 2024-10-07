import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

def neural_network() -> keras.models.Sequential:

    # Sequential model
    model = keras.models.Sequential()

    # Layers
    model.add(keras.layers.Dense(4096, input_shape=(194,), activation='relu'))
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(7, activation='sigmoid'))  # Output layer for multi-label classification

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

def kmeans() -> KMeans:
    kmeans = KMeans(n_clusters=7,
                init='random',
                tol=1e-4, 
                random_state=170,
                verbose=True)
    
    return kmeans

def cnn() -> nn.Sequential:
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25),

        nn.Flatten(),
        
        # Adjust input features based on calculated flattened size
        nn.Linear(128 * 16 * 24 , 256),  
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.Linear(256, 7)  
    )
    return model