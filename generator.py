import tensorflow as tf
import numpy as np
import os
import shutil

from random import randrange
from data import create_tmp_data


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
            self,
            features, labels,
            dim,
            batch_size=32,
            n_classes=2,
            shuffle=True
    ):

        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size

        self.labels = labels
        self.features = features
        self.example_ids = np.arange(len(self.features))

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):

        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.example_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.example_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.example_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.features[ID]

            # Store class
            y[i] = self.labels[ID]

        return X, y


