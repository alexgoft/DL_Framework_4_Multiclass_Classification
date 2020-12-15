import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
            self,
            images, labels,
            config,
            shuffle=True,
    ):

        """Initialization"""
        self.config = config

        self.dim = config['train'].get('input_size', None)
        self.batch_size = config['train'].get('batch_size', 32)

        self.labels = labels
        self.images = images
        self.example_ids = np.arange(len(self.images))

        self.n_classes = config['data'].get('num_classes', 9)

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

    def _get_sample(self, idx):

        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 4:

            # If it's training examples, we shall combine both images to a single image.
            # TODO Overlay, Median?
            image = np.average(image, axis=0)
            label /= 2

        return image, label

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim)
        # Initialization
        dim = self.images[0].shape[1:] if self.dim is None else self.dim

        images = np.empty((self.batch_size, *dim))
        labels = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            image, label = self._get_sample(ID)

            # Store sample
            images[i,] = image

            # Store class
            labels[i] = label

        return images, labels
