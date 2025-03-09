import os
import wandb
from keras.datasets import fashion_mnist


os.environ['WAND_NOTEBOOK_NAME']='ques1'

# Load Fashion-MNIST dataset
(X_train,Y_train),(X_test,Y_test) = fashion_mnist.load_data()

# Define class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Initialize wandb
wandb.init(project="DA6401-A1", entity="da24m013-iit-madras-alumni-association", name="fashion-mnist-samples-chart")

# Store one image per class
classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
sample_image=[]
label=[]
for i in range(len(X_train)):
  if(len(label)==10):
    break
  if(classes[Y_train[i]] in label):
    continue
  else:
    sample_image.append(X_train[i])
    label.append(classes[Y_train[i]])

wandb.log({"Question 1-Sample Images": [wandb.Image(img, caption=lbl) for img,lbl in zip(sample_image,label)]})