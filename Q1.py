import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Initialize wandb
wandb.init(project="DA6401-A1", name="fashion-mnist-samples-plot")

# Store one image per class
sample_images = {}
for img, label in zip(train_images, train_labels):
    if label not in sample_images:
        sample_images[label] = img
    if len(sample_images) == 10:
        break

# Create grid plot
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Sample Images", fontsize=14)

for i, (label, img) in enumerate(sample_images.items()):
    ax = axes[i // 5, i % 5]
    ax.imshow(img, cmap='gray')
    ax.set_title(class_names[label])
    ax.axis("off")

plt.tight_layout()
plt.show()

# Log plot to wandb
wandb.log({"Fashion-MNIST Samples": wandb.Image(fig)})

# End wandb run
wandb.finish()
