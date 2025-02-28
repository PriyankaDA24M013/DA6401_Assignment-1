import wandb
from Q2 import *
from config import *
def train(model, X_train, y_train, epochs=10, batch_size=64):
    wandb.init(project="DA6401-A1", name="FeedForward-training")
    
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)  # Shuffle data
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            model.forward(X_batch)  # Forward pass
            model.backward(X_batch, y_batch)  # Backpropagation

        # Compute training loss and accuracy
        predictions = np.argmax(model.forward(X_train), axis=1)
        true_labels = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == true_labels)
        loss = -np.mean(y_train * np.log(model.forward(X_train) + 1e-8))  # Cross-entropy loss

        wandb.log({"Epoch": epoch+1, "Loss": loss, "Accuracy": accuracy})
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    wandb.finish()


model = FeedforwardNeuralNetwork(layer_sizes, learning_rate)
train(model, train_images, train_labels, epochs, batch_size)