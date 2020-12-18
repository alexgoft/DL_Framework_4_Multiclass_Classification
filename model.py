import os
import matplotlib.pyplot as plt

from time import time
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from termcolor import colored


class GoftNet:

    _OPTIMIZERS = {
        'adam': Adam,
        'SGD': SGD
    }

    _LOSS_FUNCTIONS = {

        # TODO Why categorical crossentropy gives my a constant zero loss??
        'categorical_crossentropy': CategoricalCrossentropy
    }

    def __init__(self, config):

        self._num_classes = config['data']['num_classes']
        self._input_dim = config['model']['input_dim']

        # Feature Extractor Blocks
        # Assume the user didnt mess around with these arguments...
        self._num_features = config['model']['num_features']
        self._kernel_shapes = config['model']['kernel_shapes']
        self._num_conv_layers = config['model']['num_conv_layers']

        # Classifier Layers
        self._units = config['model']['units']

        # Training parameters.
        self._optimizer = config['train']['optimizer']
        self._learning_rate = config['train']['learning_rate']
        self._loss_function = config['train']['loss_function']
        self._epochs = config['train']['epochs']

        # General
        self._output_dir = config['general']['output_dir']  # Here Tensorboard logs will be written.
        self._summary = True

        # define pathes
        timestamp = str(int(time()))
        self.model_dir_path = os.path.join(self._output_dir, timestamp)
        self.model_path = os.path.join(self.model_dir_path, 'model.h5')
        self.log_dir_path = os.path.join(self._output_dir, timestamp, 'logs')

        os.makedirs(self.model_dir_path)
        os.makedirs(self.log_dir_path)

        self._create_model()

    def load_model(self, path):
        self._model = load_model(path)
        self._compile()

    def _create_block(self, num_features, kernel_shape, number_conv_layers,
                      first_layer=False, last_layer=False, padding='same'):

        for _ in range(number_conv_layers):

            if first_layer:
                self._model.add(Conv2D(num_features, kernel_shape, padding=padding, input_shape=self._input_dim))
            else:
                self._model.add(Conv2D(num_features, kernel_shape, padding=padding))

            self._model.add(BatchNormalization())
            self._model.add(Activation('relu'))

        if last_layer:
            self._model.add(MaxPooling2D(pool_size=(2, 2)))

    def _compile(self):

        optimizer = self._OPTIMIZERS[self._optimizer](lr=self._learning_rate)
        loss_function = self._LOSS_FUNCTIONS[self._loss_function]()

        self._model.compile(
            loss=loss_function,
            optimizer=optimizer,

            metrics=['accuracy']

        )

    def _create_model(self, print_color='yellow'):

        print(colored('###################################', print_color))
        print(colored('######### CREATING MODEL #########', print_color))
        print(colored('###################################', print_color))

        # build the CNN architecture with Keras Sequential API
        self._model = Sequential()

        # ---------------------------------------- #
        # --------- DEFINE F.E BLOCKS ------------ #
        # ---------------------------------------- #
        for i, (num_features, kernel_shape, num_conv_layers) in enumerate(zip(self._num_features,
                                                                              self._kernel_shapes,
                                                                              self._num_conv_layers)):
            if i == 0:  # First Layer. Need to define input layer
                self._create_block(num_features, kernel_shape, num_conv_layers, first_layer=True)

            elif i == len(self._num_features):  # Last Layer. No max pooling.
                pass

            else:
                self._create_block(num_features, kernel_shape, num_conv_layers, last_layer=True)

        # ---------------------------------------- #
        # ----- DEFINE CLASSIFIER BLOCKS --------- #
        # ---------------------------------------- #
        self._model.add(Flatten())

        for units in self._units:
            self._model.add(Dense(units))
            self._model.add(Activation('relu'))

        self._model.add(Dense(self._num_classes))
        self._model.add(Activation('softmax'))

        # Compile the model with chosen optimizer.
        self._compile()

        # Summary
        if self._summary:
            self._model.summary()

    def train(self, train_data, val_data):

        callbacks = [
            ModelCheckpoint(filepath=self.model_path, save_best_only=True, monitor='val_loss'),
            TensorBoard(log_dir=self.log_dir_path)
        ]

        train_log = self._model.fit_generator(

            generator=train_data,
            validation_data=val_data,

            epochs=self._epochs,

            callbacks=callbacks,

            verbose=1
        )

        self.plot_log(train_log=train_log, model_dir_path=self.model_dir_path)

    @staticmethod
    def plot_log(train_log, model_dir_path):

        # Plot training & validation accuracy values
        f = plt.figure(1)
        plt.plot(train_log.history['accuracy'])
        plt.plot(train_log.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        f.savefig(os.path.join(model_dir_path, 'acc.png'))

        # Plot training & validation loss values
        g = plt.figure(2)
        plt.plot(train_log.history['loss'])
        plt.plot(train_log.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        g.savefig(os.path.join(model_dir_path, 'loss.png'))
