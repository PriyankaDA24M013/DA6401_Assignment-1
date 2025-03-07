import numpy as np
from activation import get_activation
from utils import start_weights_and_bias

class FeedForwardNN:
    def __init__(self, input_size, num_layers, hidden_size, activation, weight_init='random'):
        self.num_layers = num_layers
        self.activation, self.activation_derivative = get_activation(activation)  # Retrieve activation function and its derivative

        # Define layer sizes: [input_size] -> hidden layers -> output layer (10 neurons for classification)
        layer_sizes = [input_size] + [hidden_size] * num_layers + [10]
        n_layers = [(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

        # Initialize weights and biases using start_weights_and_bias
        self.weights, self.biases = start_weights_and_bias(n_layers, weight_init)

        # Initialize gradients with zero
        self.dW = [np.zeros_like(w) for w in self.weights]  # Gradients for weights
        self.dB = [np.zeros_like(b) for b in self.biases]   # Gradients for biases

    def forward(self, X):
        self.a = [X]  # Store activations (input layer)
        
        # Forward pass through each layer
        for i in range(self.num_layers + 1):  # +1 includes the output layer
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]  # Compute weighted sum
            if i < self.num_layers:
                a = self.activation(z)  # Apply activation for hidden layers
            else:
                a = softmax(z)  # Use softmax for the output layer
            
            self.a.append(a)  # Store activation
        
        return self.a[-1]  # Return output of the final layer

    def backward(self, y_true):
        m = y_true.shape[0]  # Number of samples in the batch
        dz = self.a[-1] - y_true  # Compute gradient of loss with respect to final layer output (softmax derivative)

        # Iterate backward through all layers (from output to input layer)
        for i in reversed(range(self.num_layers + 1)):
            self.dW[i] = np.dot(self.a[i].T, dz) / m  # Compute gradient of weights
            self.dB[i] = np.sum(dz, axis=0, keepdims=True) / m  # Compute gradient of biases
            
            # Compute dz for the previous layer (skip for input layer)
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(self.a[i]) 

    def parameters(self):
        return self.weights, self.biases, self.dW, self.dB

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # Numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)