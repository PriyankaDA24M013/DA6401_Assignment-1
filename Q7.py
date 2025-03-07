import wandb
import numpy as np
from utils import test_accuracy
from dataset import load_data
from model import FeedForwardNN

# Step 1: Retrieve the best run from W&B
api = wandb.Api()
sweep = api.sweep("da24m013-iit-madras-alumni-association/DA6401-A1/gnbubve1")
runs = sorted(sweep.runs,
  key=lambda run: run.summary.get("Val Accuracy", 0), reverse=True)

best_run = runs[0]
best_params = best_run.config

# Step 2: Load the test dataset
_, _, X_test, y_test = load_data("fashion_mnist")
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Step 3: Load the model with best hyperparameters
model = FeedForwardNN(
    input_size=784,  # Assuming flattened 28x28 images
    num_layers=best_params["num_layers"],
    hidden_size=best_params["hidden_size"],
    activation=best_params["activation"],
    weight_init=best_params["weight_init"]
)

# Step 4: Download and load model weights
weights_data = np.load("model_weights.npy", allow_pickle=True).item()
model.weights = weights_data["weights"]
model.biases = weights_data["biases"]

# Step 5: Compute predictions
y_pred = model.forward(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to labels




# Convert y_test from one-hot encoding to integer labels if necessary
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)  # Convert from one-hot to class indices
    
# print("y_pred shape:", y_pred_labels.shape)
# print("y_test shape:", y_test.shape)

# Compute accuracy
test_accuracy = test_accuracy(y_pred_labels,y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 6: Log confusion matrix in W&B
wandb.init(project='DA6401-A1', name="best_model_evaluation")
cm = wandb.plot.confusion_matrix(
  y_true=y_test,
  preds=y_pred_labels,
  class_names=class_names
)
wandb.log({"confusion_matrix": cm})
wandb.finish()
