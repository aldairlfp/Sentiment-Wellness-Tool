import tensorflow as tf
from tensorflow import keras
import numpy as np 
from sklearn.cluster import KMeans

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

def cnn():
    pass