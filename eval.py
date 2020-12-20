import yaml

from data import get_data
from model import GoftNet


def main():

    config = yaml.safe_load(open("config.yaml", 'r'))

    # =========================================== #
    # =============== PREPARE DATA ============== #
    # =========================================== #
    x_train, y_train, x_test, y_test = get_data(config=config)

    # =========================================== #
    # =========== TEST MODEL ON DATA ============ #
    # =========================================== #
    model = GoftNet(config=config)
    model.load_model()

    # =========================================== #
    # ================ EVALUATE ================= #
    # =========================================== #
    y_pred = model.inference_on_data(test_data=x_test)
    model.print_metrics(y_pred, y_test)


if __name__ == '__main__':
    main()
