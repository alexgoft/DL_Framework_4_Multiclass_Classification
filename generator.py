import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
            self,
            images, labels,
            config,
            gen_type,
            shuffle=True,
    ):

        """Initialization"""
        self.config = config

        self._gen_type = gen_type

        self._dim = config['model'].get('input_dim', None)
        self._batch_size = config['train'].get('batch_size', 32)

        self._labels = labels
        self._images = images
        self._example_ids = np.arange(len(self._images))

        self._n_classes = config['data'].get('num_classes', 9)

        self._shuffle = shuffle

        self._indexes = None

        self.on_epoch_end()

    def __len__(self):

        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self._example_ids) / self._batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self._batch_size:(index + 1) * self._batch_size]

        # Find list of IDs
        list_IDs_temp = [self._example_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self._example_ids))
        if self._shuffle:
            np.random.shuffle(self.indexes)

    def _get_sample(self, idx):

        image = self._images[idx]
        label = self._labels[idx]

        if len(image.shape) == 4:

            # If it's training examples, we shall combine both images to a single image.
            # TODO Overlay, Median?
            image = np.average(image, axis=0)
            label /= 2

        return image, label

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim)

        # Initialization
        if self._dim is None:
            dim = self._images[0].shape
            dim = dim if len(dim) == 3 else dim[1:]
        else:
            dim = self._dim

        images = np.empty((self._batch_size, *dim))
        labels = np.empty((self._batch_size, self._n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            image, label = self._get_sample(ID)

            # Store sample
            images[i,] = image

            # Store class
            labels[i] = label

        return images, labels
