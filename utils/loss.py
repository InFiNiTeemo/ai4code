import numpy as np


def mcrmse(y, y_hat, n_class):
    """
    :param y: 1-d array
    :param y_hat: 1-d array
    """
    y = y.reshape(-1, n_class)
    y_hat = y_hat.reshape(-1, n_class)

    return np.mean(np.sqrt(np.mean((y_hat - y) ** 2, axis=0)))