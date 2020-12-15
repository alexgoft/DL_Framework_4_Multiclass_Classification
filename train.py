import yaml
from tensorflow.python.keras.models import Sequential

from data import get_data
from generator import DataGenerator


def main():
    config = yaml.safe_load(open("config.yaml", 'r'))

    epochs = config['train']['epochs']
    optimizer = config['train']['optimizer']
    loss_function = config['train']['loss_function']

    # ------------------------------------------------- #

    # Create data for training/validation
    train_x, train_y, val_x, val_y = get_data(config=config)

    train_generator = DataGenerator(images=train_x, labels=train_y, config=config)
    val_generator = DataGenerator(images=val_x, labels=val_y, config=config)

    # Create model
    model = Sequential()
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    # Run training loop.
    model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs, verbose=0)


if __name__ == '__main__':
    main()
