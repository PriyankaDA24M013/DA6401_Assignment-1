import numpy as np

def cross_entropy(y_pred, y_true):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Prevent log(0)
    return -np.sum(y_true * np.log(y_pred)) / m

def cross_entropy_derivative(y_pred, y_true):
    return y_pred - y_true

def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mean_squared_error_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def get_loss(name):
    losses = {
        'cross_entropy': (cross_entropy, cross_entropy_derivative),
        'mean_squared_error': (mean_squared_error, mean_squared_error_derivative)
    }
    return losses.get(name, (cross_entropy, cross_entropy_derivative))
