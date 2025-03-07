# DA6401_Assignment-1

**Q1.py** contains the code to plot the fashion_mnist data samples of each class

**Q7_cn** plots the confusion matrix and prints test accuracy on trained model which is saved as _model_weights.npy_

The flow of the code is as follows:

1. Run _sweep.py_ to find the best hyperparameters.
2. Use the best hyperparameters to train the model for more epochs using _train.py_

Supporting Files:

1. dataset.py: Loads and preprocesses the dataset.

2. activation.py: Provides activation functions and their derivatives.

3. loss.py: Implements loss functions and their derivatives.

4. model.py: Defines the FeedForwardNN class for the neural network.

5. optimizer.py: Implements various optimizers (SGD, Momentum, NAG, RMSprop, Adam, Nadam).

6. utils.py: Contains utility functions like accuracy, one_hot_encode, and initialize_weights

WandB Report Link - https://wandb.ai/da24m013-iit-madras-alumni-association/DA6401-A1/reports/DA24M013-Assignment-1-Report--VmlldzoxMTU3NTYyNg

Github Repo Link - https://github.com/PriyankaDA24M013/DA6401_Assignment-1.git
