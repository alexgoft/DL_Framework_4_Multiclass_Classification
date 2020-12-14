import yaml
import numpy as np
import random

from itertools import combinations
from argparse import Namespace
from tensorflow.keras.datasets import cifar10


def get_class_examples(data, labels, cls, num_examples):
    cls_examples_idx = list(np.where(labels == cls)[0])
    cls_sampled_example_idx = random.sample(cls_examples_idx, num_examples)
    cls_examples = data[cls_sampled_example_idx]

    return cls_examples


def tuple_to_prob_vector(num_classes, class_1, class_2):
    classes_eye = np.eye(num_classes)

    label = classes_eye[class_1] + classes_eye[class_2]
    # label /= 2

    return label


def normalize_data(data):
    return np.array(data) / 255


def labels_to_one_hot(num_classes, labels):
    return np.array([np.eye(num_classes)[label] for label in labels])


def get_data(config):
    num_classes = config['data']['num_classes']

    train_examples_num = config['data']['train_set_size']
    val_examples_num = config['data']['val_set_size']

    # ---------------------------------- #
    # --------- GET RAW DATA ----------- #
    # ---------------------------------- #
    (train_x_raw, train_y_raw), (val_x_raw, val_y_raw) = cifar10.load_data()

    train_y_raw = np.squeeze(train_y_raw)
    val_y_raw = np.squeeze(val_y_raw)

    # ---------------------------------- #
    # ------------ TRAIN --------------- #
    # ---------------------------------- #
    # TODO Given solution creates 49,995 train examples..

    # Create all tuple combinations given the class number.
    class_combinations = set(combinations(np.arange(num_classes), 2))

    examples_per_duo = train_examples_num // len(class_combinations)

    train_x = []
    train_y = []
    for cls_1, cls_2 in class_combinations:
        cls_1_examples = get_class_examples(train_x_raw, train_y_raw, cls=cls_1, num_examples=examples_per_duo)
        cls_2_examples = get_class_examples(train_x_raw, train_y_raw, cls=cls_2, num_examples=examples_per_duo)

        curr_comb_label = tuple_to_prob_vector(num_classes=num_classes, class_1=cls_1, class_2=cls_2)
        curr_comb_labels = [curr_comb_label] * examples_per_duo

        curr_comb_data = list(zip(cls_1_examples, cls_2_examples))

        train_x += curr_comb_data
        train_y += curr_comb_labels

    train_x = normalize_data(data=train_x)
    train_y = np.array(train_y)

    # ---------------------------------- #
    # -------------- VAL --------------- #
    # ---------------------------------- #
    # Sample needed number of examples
    val_x = np.array(random.sample(list(val_x_raw), val_examples_num))

    # Normalize images to be between 0 and 1
    val_x = normalize_data(data=val_x)

    # validation labels to one_hot
    val_y = labels_to_one_hot(num_classes=num_classes, labels=val_y_raw)

    print('DATA SHAPES:')
    print('\tTRAIN X', train_x.shape)
    print('\tTRAIN Y', train_y.shape)
    print('\tVAL X', val_x.shape)
    print('\tVAL Y', val_y.shape)  # Val is a "two-hot-vector". It will be changed to a prob vector according to alpha.

    return train_x, train_y, val_x, val_y


def test():
    config = yaml.safe_load(open("config.yaml", 'r'))

    train_x, train_y, val_x, val_y = get_data(config=config)


if __name__ == '__main__':
    test()