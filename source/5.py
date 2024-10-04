import tensorflow as tf
from tensorflow import keras
import numpy as np
r_data = np.random.rand(1000, 194)
r_labels = np.random.randint(0, 7, size=(len(r_data),))  


def create_model():
    # Create a Sequential model
    model = keras.models.Sequential()

    # Add layers
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

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model