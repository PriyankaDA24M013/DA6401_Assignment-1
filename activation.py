import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow in exp()
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

def get_activation(name):
    activations = {
        'sigmoid': (sigmoid, sigmoid_derivative),
        'tanh': (tanh, tanh_derivative),
        'ReLU': (relu, relu_derivative),
        'identity': (identity, identity_derivative)
    }
    return activations.get(name, (sigmoid, sigmoid_derivative))