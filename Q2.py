import numpy as np
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to [0,1]
train_images = train_images.reshape(-1, 784) / 255.0
test_images = test_images.reshape(-1, 784) / 255.0

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Print shapes
print(f"Train Images: {train_images.shape}, Train Labels: {train_labels.shape}")
print(f"Test Images: {test_images.shape}, Test Labels: {test_labels.shape}")

# ReLU activation and derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax function for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initializes a neural network.
        :param layer_sizes: List containing sizes of each layer, e.g., [784, 128, 64, 10]
        :param learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        # Initialize weights and biases (small random values)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]

    def forward(self, X):
        """
        Forward pass through the network.
        :param X: Input data (batch_size, input_dim)
        :return: Output probabilities
        """
        self.activations = [X]  # Store activations for backprop
        self.z_values = []  # Store linear outputs

        for i in range(len(self.weights) - 1):  # Hidden layers
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))

        # Output layer (Softmax activation)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        return output

    def backward(self, X, y):
        """
        Backpropagation to update weights and biases.
        :param X: Input data (batch_size, input_dim)
        :param y: One-hot encoded labels (batch_size, num_classes)
        """
        m = X.shape[0]  # Batch size
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Compute the loss gradient (Softmax cross-entropy derivative)
        dz = self.activations[-1] - y  # (batch_size, num_classes)

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(self.activations[i].T, dz) / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:  # Compute dz for next layer
                dz = np.dot(dz, self.weights[i].T) * relu_derivative(self.z_values[i-1])

        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]
