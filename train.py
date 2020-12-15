import yaml
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from data import get_data
from generator import DataGenerator


def main():
    config = yaml.safe_load(open("config.yaml", 'r'))

    # train
    epochs = config['train']['epochs']
    optimizer = config['train']['optimizer']
    loss_function = config['train']['loss_function']

    # data
    num_classes = config['data']['num_classes']

    # model
    input_dim = eval(config['model']['input_dim'])

    # ------------------------------------------------- #

    # Create data for training/validation
    train_x, train_y, val_x, val_y = get_data(config=config)

    train_generator = DataGenerator(images=train_x, labels=train_y, config=config)
    val_generator = DataGenerator(images=val_x, labels=val_y, config=config)

    # Create model
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    # Run training loop.
    model.fit_generator(generator=val_generator, validation_data=val_generator, epochs=epochs, verbose=0)


if __name__ == '__main__':
    main()
