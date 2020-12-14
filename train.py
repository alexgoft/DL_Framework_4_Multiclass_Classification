import yaml

from data import get_data
from generator import DataGenerator


def main():
    config = yaml.safe_load(open("config.yaml", 'r'))

    train_x, train_y, val_x, val_y = get_data(config=config)

    train_generator = DataGenerator(images=train_x, labels=train_y, config=config)
    val_generator = DataGenerator(images=val_x, labels=val_y, config=config)



if __name__ == '__main__':
    main()
