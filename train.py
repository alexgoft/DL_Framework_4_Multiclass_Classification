import yaml

from data import get_data
from generator import DataGenerator
from model import GoftNet


def main():

    config = yaml.safe_load(open("config.yaml", 'r'))

    # =========================================== #
    # =============== PREPARE DATA ============== #
    # =========================================== #
    train_x, train_y, val_x, val_y = get_data(config=config)

    train_generator = DataGenerator(images=train_x, labels=train_y, config=config, gen_type='train')
    val_generator = DataGenerator(images=val_x, labels=val_y, config=config, gen_type='val')

    # =========================================== #
    # =============== CREATE MODEL ============== #
    # =========================================== #
    model = GoftNet(config=config)

    # =========================================== #
    # =============== TRAIN MODEL =============== #
    # =========================================== #
    model.train(train_data=train_generator, val_data=val_generator)


if __name__ == '__main__':
    main()
