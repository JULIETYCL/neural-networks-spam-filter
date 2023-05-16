# Spam Classifier
There are many powerful libraries available, such as PyTorch and TensorFlow, for building neural networks. However, it's essential to understand the underlying principles that guide their operation. In this repository, we aim to elucidate these principles by providing a simple neural network-based spam classifier, implemented from scratch in Python. This model has been trained on a dataset containing emails that have been meticulously labeled as 'spam' or 'ham' (non-spam).

## Installation
Before running the script, make sure you have `numpy` installed in your Python environment. To install it, you can use the following command:
```
pip install numpy
```
## Usage
The main script `spam_classifier.py` contains the implementation of the spam classifier. It consists of the following parts:

1. Loading and processing the data from CSV files.
2. Defining the neural network architecture and training using gradient descent.
3. Testing the classifier and calculating the accuracy.

To run the script, simply execute the following command:

```
python spam_classifier.py
```

After running the script, you will see the accuracy improving as the model trains. Once the training is complete, the script will display the final accuracy of the classifier on the test data.

## Model Overview
The spam classifier uses a simple feedforward neural network with one hidden layer. The network architecture consists of 54 input nodes, 20 hidden nodes, and a single output node. The sigmoid activation function is used for both hidden and output layers.

The model is trained using gradient descent, with a learning rate of 0.1. The training data is divided into 10 batches, and the model is trained for 100 epochs. During training, the script prints the accuracy of the classifier on the current batch.

After training, the classifier is tested on the test data, and the overall accuracy is calculated and displayed.
