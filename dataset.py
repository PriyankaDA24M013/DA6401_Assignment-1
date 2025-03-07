import numpy as np
from keras.datasets import fashion_mnist, mnist
from utils import one_hot_encode

def load_data(dataset_name):
    if dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
    y_train, y_test = one_hot_encode(y_train), one_hot_encode(y_test)
    
    return X_train, y_train, X_test, y_test
