# CS6910-Assignment-1
#### February 28, 2023

This repo is my implementation of a deep neural network from first principles as part of Assignment-01 of CS6910.

## Code organization:

### ANN implementation

*ann.py*: Contains implementation of the ANN with backpropagation & forward propagation.

*ann_utils.py*: Contains activation functions & functions for parameter initialization.

*gd.py*: Contains functions for gradient descent.

### Training

*train.py*: Used to train the ANN. Takes parameters as specified by the problem statement. Creates one run in the `projectname`.

*train_utils.py*: Contains function to get accuracy and loss.

*maps.py*: Contains mapping from parameter to functions and variables. It is used by *train.py*.

*sweep.ipynb*: Wandb sweep for fashion-MNIST dataset with cross-entropy loss.

*sweep_mse.ipynb*: Wandb sweep for fashion-MNIST dataset with mean-squared error loss.

### Question specific code:

*Q1_sample_images.ipynb*

*Q7_confusion_mat.ipynb*

*Q10_mnist.ipynb*