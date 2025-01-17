import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from time import time
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy, MSE
from termcolor import colored
from sklearn.metrics import confusion_matrix, classification_report


class GoftNet:
    _OPTIMIZERS = {
        'adam': Adam,
        'SGD': SGD,
        'RMSprop': RMSprop
    }

    _LOSS_FUNCTIONS = {

        # TODO Why categorical crossentropy gives my a constant zero loss??
        'categorical_crossentropy': CategoricalCrossentropy(),
        'binary_crossentropy': BinaryCrossentropy(),
        'mse': MSE,
    }

    def __init__(self, config):

        self._num_classes = config['data']['num_classes']
        self._class_labels = config['data']['labels']
        self._class_labels_dict = {i: self._class_labels[i] for i in range(len(self._class_labels))}
        self._input_dim = config['model']['input_dim']

        # Feature Extractor Blocks
        # Assume the user didnt mess around with these arguments...
        self._num_features = config['model']['num_features']
        self._kernel_shapes = config['model']['kernel_shapes']
        self._num_conv_layers = config['model']['num_conv_layers']

        # Classifier Layers
        self._units = config['model']['units']
        self._last_layer_activation = config['model'].get('last_later_activation', None)

        # Training parameters.
        self._optimizer = config['train']['optimizer']
        self._loss_function = config['train']['loss_function']
        self._epochs = config['train']['epochs']

        # General
        self._output_dir = config['general']['output_dir']  # Here Tensorboard logs will be written.
        self._summary = True

        # Eval
        self._model_path = config['eval']['model_path']

        # define pathes
        timestamp = str(int(time()))
        self.model_dir_path = os.path.join(self._output_dir, timestamp)
        self.model_path = os.path.join(self.model_dir_path, 'model.h5')
        self.log_dir_path = os.path.join(self._output_dir, timestamp, 'logs')

        os.makedirs(self.model_dir_path, exist_ok=True)
        os.makedirs(self.log_dir_path, exist_ok=True)

        self._create_model()

    def _create_block(self,
                      num_features, kernel_shape, number_conv_layers,
                      first_layer, last_layer,
                      padding='same'
                      ):

        for _ in range(number_conv_layers):

            if first_layer:
                self._model.add(Conv2D(num_features, kernel_shape, padding=padding, input_shape=self._input_dim))
            else:
                self._model.add(Conv2D(num_features, kernel_shape, padding=padding))

            self._model.add(BatchNormalization())
            # self._model.add(Dropout(0.3))
            self._model.add(Activation('relu'))

        if not last_layer:
            self._model.add(MaxPooling2D(pool_size=(2, 2)))

    def _get_optimizer(self):
        name = self._optimizer['name']
        opt_params = self._optimizer['params']

        return self._OPTIMIZERS[name](**opt_params)

    def _compile(self):

        optimizer = self._get_optimizer()
        loss_function = self._LOSS_FUNCTIONS[self._loss_function]

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
                first_layer, last_layer = True, False

            elif i == len(self._num_features) - 1:  # Last Layer. No max pooling.
                first_layer, last_layer = False, True

            else:
                first_layer, last_layer = False, False

            self._create_block(
                num_features, kernel_shape, num_conv_layers,
                first_layer=first_layer, last_layer=last_layer
            )

        # ---------------------------------------- #
        # ----- DEFINE CLASSIFIER BLOCKS --------- #
        # ---------------------------------------- #
        self._model.add(Flatten())

        for units in self._units:
            self._model.add(Dense(units))
            self._model.add(Activation('relu'))

        self._model.add(Dense(self._num_classes))
        if self._last_layer_activation is not None:
            # If last layer activation is None, loss wil be calculated directly on logits.
            self._model.add(Activation(self._last_layer_activation))

        # Compile the model with chosen optimizer.
        self._compile()

        # Summary
        if self._summary:
            self._model.summary()

    def train(self, train_data, val_data):

        callbacks = [
            ModelCheckpoint(filepath=self.model_path, monitor='val_accuracy', mode='max', save_best_only=True),
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

    def load_model(self, ):
        self._model = load_model(self._model_path)
        self._compile()

    def inference_on_data(self, test_data):
        results = self._model.predict(test_data, verbose=1)
        results = [np.eye(self._num_classes)[np.argmax(res)] for res in results]  # Turn results to one hot.
        return results

    def print_metrics(self, y_pred, y_test):

        y_pred_labels = [self._class_labels_dict[class_num] for class_num in np.argmax(y_pred, axis=1)]
        y_test_labels = [self._class_labels_dict[class_num] for class_num in np.argmax(y_test, axis=1)]

        cm = confusion_matrix(y_pred_labels, y_test_labels, labels=np.unique(y_test_labels))
        cm = pd.DataFrame(cm, index=np.unique(y_test_labels), columns=np.unique(y_test_labels))

        report = classification_report(y_test, y_pred)

        print(colored("\n===================================================", 'yellow'))
        print(colored("============== CLASSIFICATION REPORT ==============", 'yellow'))
        print(colored("===================================================", 'yellow'))
        print(colored(report + '\n', 'yellow'))
        print(colored(cm, 'yellow'))

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
