import numpy as np

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        weights, biases, dW, dB = self.params
        for i in range(len(weights)):
            weights[i] -= self.lr * dW[i]
            biases[i] -= self.lr * dB[i]

class Momentum:
    def __init__(self, params, lr, momentum):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.vW = [np.zeros_like(w) for w in params[0]]
        self.vB = [np.zeros_like(b) for b in params[1]]
    
    def step(self):
        weights, biases, dW, dB = self.params
        for i in range(len(weights)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW[i]
            self.vB[i] = self.momentum * self.vB[i] - self.lr * dB[i]
            weights[i] += self.vW[i]
            biases[i] += self.vB[i]

class NAG:
    def __init__(self, params, lr, momentum):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.vW = [np.zeros_like(w) for w in params[0]]
        self.vB = [np.zeros_like(b) for b in params[1]]
    
    def step(self):
        weights, biases, dW, dB = self.params
        for i in range(len(weights)):
            lookahead_w = weights[i] + self.momentum * self.vW[i]
            lookahead_b = biases[i] + self.momentum * self.vB[i]
            
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW[i]
            self.vB[i] = self.momentum * self.vB[i] - self.lr * dB[i]
            weights[i] = lookahead_w + self.vW[i]
            biases[i] = lookahead_b + self.vB[i]

class RMSprop:
    def __init__(self, params, lr, beta, epsilon):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.sW = [np.zeros_like(w) for w in params[0]]
        self.sB = [np.zeros_like(b) for b in params[1]]
    
    def step(self):
        weights, biases, dW, dB = self.params
        for i in range(len(weights)):
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (dW[i] ** 2)
            self.sB[i] = self.beta * self.sB[i] + (1 - self.beta) * (dB[i] ** 2)
            weights[i] -= self.lr * dW[i] / (np.sqrt(self.sW[i]) + self.epsilon)
            biases[i] -= self.lr * dB[i] / (np.sqrt(self.sB[i]) + self.epsilon)

class Adam:
    def __init__(self, params, lr, beta1, beta2, epsilon):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mW = [np.zeros_like(w) for w in params[0]]
        self.mB = [np.zeros_like(b) for b in params[1]]
        self.vW = [np.zeros_like(w) for w in params[0]]
        self.vB = [np.zeros_like(b) for b in params[1]]
        self.t = 0
    
    def step(self):
        weights, biases, dW, dB = self.params
        self.t += 1
        for i in range(len(weights)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW[i]
            self.mB[i] = self.beta1 * self.mB[i] + (1 - self.beta1) * dB[i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.vB[i] = self.beta2 * self.vB[i] + (1 - self.beta2) * (dB[i] ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            mB_hat = self.mB[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            vB_hat = self.vB[i] / (1 - self.beta2 ** self.t)

            weights[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
            biases[i] -= self.lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)

class Nadam:
    def __init__(self, params, lr, beta1, beta2, epsilon):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mW = [np.zeros_like(w) for w in params[0]]
        self.mB = [np.zeros_like(b) for b in params[1]]
        self.vW = [np.zeros_like(w) for w in params[0]]
        self.vB = [np.zeros_like(b) for b in params[1]]
        self.t = 0
    
    def step(self):
        weights, biases, dW, dB = self.params
        self.t += 1
        for i in range(len(weights)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * dW[i]
            self.mB[i] = self.beta1 * self.mB[i] + (1 - self.beta1) * dB[i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.vB[i] = self.beta2 * self.vB[i] + (1 - self.beta2) * (dB[i] ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            mB_hat = self.mB[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            vB_hat = self.vB[i] / (1 - self.beta2 ** self.t)

            mW_nadam = self.beta1 * mW_hat + (1 - self.beta1) * dW[i] / (1 - self.beta1 ** self.t)
            mB_nadam = self.beta1 * mB_hat + (1 - self.beta1) * dB[i] / (1 - self.beta1 ** self.t)

            weights[i] -= self.lr * mW_nadam / (np.sqrt(vW_hat) + self.epsilon)
            biases[i] -= self.lr * mB_nadam / (np.sqrt(vB_hat) + self.epsilon)

def get_optimizer(name, params, args):
    optimizers = {
        'sgd': SGD(params, args.learning_rate),
        'momentum': Momentum(params, args.learning_rate, args.momentum),
        'nag': NAG(params, args.learning_rate, args.momentum),
        'rmsprop': RMSprop(params, args.learning_rate, args.beta, args.epsilon),
        'adam': Adam(params, args.learning_rate, args.beta1, args.beta2, args.epsilon),
        'nadam': Nadam(params, args.learning_rate, args.beta1, args.beta2, args.epsilon)
    }
    return optimizers.get(name, SGD(params, args.learning_rate))
