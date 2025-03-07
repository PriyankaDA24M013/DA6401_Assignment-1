import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def start_weights_and_bias(n_layers, weight_init):
    """Initialize weights and biases based on the specified method."""
    initial_weights = []
    initial_bias = []

    for i in range(len(n_layers)):
        input_size, hidden_size = n_layers[i]  # Unpack input and output sizes
        if weight_init == "random":
            w = np.random.uniform(-1, 1, (input_size, hidden_size))  # Shape: (input_size, hidden_size)
            b = np.random.rand(1, hidden_size)  # Shape: (1, hidden_size)
        elif weight_init == "Xavier":
            num = np.sqrt(6 / (input_size + hidden_size))
            w = np.random.uniform(-num, num, (input_size, hidden_size))  # Shape: (input_size, hidden_size)
            b = np.random.uniform(-num, num, (1, hidden_size))  # Shape: (1, hidden_size)
        else:
            raise ValueError("Invalid weight initialization method. Use 'random' or 'Xavier'.")

        initial_weights.append(w)
        initial_bias.append(b)

    return initial_weights, initial_bias

# Compute accuracy
def test_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

