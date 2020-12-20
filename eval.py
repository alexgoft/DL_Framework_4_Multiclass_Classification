import yaml

from data import get_data
from model import GoftNet


def main():
    MODEL_PATH = '/hdd/cool_cifar10/output/1608485588/model.h5'

    config = yaml.safe_load(open("config.yaml", 'r'))

    # =========================================== #
    # =============== PREPARE DATA ============== #
    # =========================================== #
    x_train, y_train, x_test, y_test = get_data(config=config)

    # =========================================== #
    # =========== TEST MODEL ON DATA ============ #
    # =========================================== #
    model = GoftNet(config=config)
    model.load_model(path=MODEL_PATH)

    y_pred = model.inference_on_data(test_data=x_test)
    cm, metrics = model.get_metrics(y_pred, y_test)

    print("\n===================================================")
    print("============== CLASSIFICATION REPORT ==============")
    print("===================================================")
    print(metrics, '\n')
    print(cm)


if __name__ == '__main__':
    main()
