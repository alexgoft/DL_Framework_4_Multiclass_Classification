import yaml

from data import get_data
from generator import DataGenerator
from model import GoftNet


def main():
    """
    Training data somewhat reminds me MixUp augmentation...
    https://towardsdatascience.com/2-reasons-to-use-mixup-when-training-yor-deep-learning-models-58728f15c559
    """
    config = yaml.safe_load(open("config.yaml", 'r'))

    # Create data for training/validation
    train_x, train_y, val_x, val_y = get_data(config=config)

    train_generator = DataGenerator(images=train_x, labels=train_y, config=config, gen_type='train')
    val_generator = DataGenerator(images=val_x, labels=val_y, config=config, gen_type='val')

    # Create model
    model = GoftNet(config=config)
    model.train(train_data=train_generator, val_data=val_generator)


if __name__ == '__main__':
    main()
