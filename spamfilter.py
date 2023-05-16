import numpy as np
from typing import Tuple


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training and testing data from CSV files.
    Returns:
        Tuple containing training and testing data as numpy arrays.
    """
    training_data = np.loadtxt(
        open("data/training_spam.csv"), delimiter=",", dtype=np.int32)
    testing_data = np.loadtxt(
        open("data/testing_spam.csv"), delimiter=",", dtype=np.int32)

    return training_data, testing_data


def create_batches(training_data: np.ndarray, num_batches: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create feature sets and label sets from the training data.

    Args:
        training_data: numpy array containing the training data.
        num_batches: number of batches to create.

    Returns:
        Tuple containing feature sets and label sets as numpy arrays.
    """

    feature_sets = np.array([training_data[i*100:(i+1)*100, 1:]
                            for i in range(num_batches)])
    label_sets = np.array([training_data[i*100:(i+1)*100, :1]
                          for i in range(num_batches)])

    return feature_sets, label_sets


class SpamClassifier:
    def __init__(self):
        self.learning_rate = 0.1
        self.input_nodes = 54
        self.hidden_nodes = 20

        self.weights_input_hidden = np.random.uniform(
            -1, 1, (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_output = np.random.uniform(
            -1, 1, (self.hidden_nodes, 1))
        self.bias_hidden = np.random.uniform(-1, 1, (self.hidden_nodes, 1))
        self.bias_output = np.random.uniform(-1, 1, (1, 1))

    @staticmethod
    def sigmoid(layer_input: np.ndarray) -> np.ndarray:
        """
       Calculate the sigmoid activation function for the given input.

       Args:
           layer_input: numpy array of input values.

       Returns:
           numpy array containing the sigmoid activation values.
       """
        return 1.0 / (1.0 + np.exp(-layer_input))

    @staticmethod
    def derivative_sigmoid(sigmoid_output: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the sigmoid activation function.

        Args:
            sigmoid_output: numpy array of sigmoid activation values.

        Returns:
            numpy array containing the derivative of the sigmoid activation values.
        """
        return sigmoid_output * (1.0 - sigmoid_output)

    def forward_propagation(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform forward propagation through the neural network.

        Args:
            input_data: numpy array containing the input data.

        Returns:
            Tuple containing the hidden layer input, hidden layer output, output layer input, and output layer output.
        """
        hidden_layer_input = input_data.dot(
            self.weights_input_hidden) + np.transpose(self.bias_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = hidden_layer_output.dot(
            self.weights_hidden_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output

    def backward_propagation(self, hidden_layer_input: np.ndarray, hidden_layer_output: np.ndarray, output_layer_output: np.ndarray, input_data: np.ndarray, true_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the backpropagation step for the neural network.

        Args:
            hidden_layer_input (np.ndarray): The input to the hidden layer.
            hidden_layer_output (np.ndarray): The output of the hidden layer.
            output_layer_output (np.ndarray): The output of the output layer.
            input_data (np.ndarray): The input data used for forward propagation.
            true_values (np.ndarray): The true values for the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Gradients of the weights and biases (weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update).
        """
        m = output_layer_output.size
        output_layer_error = output_layer_output - true_values
        weights_hidden_output_update = (
            1 / m) * hidden_layer_output.T.dot(output_layer_error)
        bias_output_update = (1 / m) * np.sum(output_layer_error)
        hidden_layer_error = self.weights_hidden_output * \
            self.derivative_sigmoid(self.sigmoid(
                hidden_layer_input)).T.dot(output_layer_error)
        weights_input_hidden_update = input_data.T.dot(
            (1 / m) * self.weights_hidden_output.T * self.derivative_sigmoid(self.sigmoid(hidden_layer_input)) * output_layer_error)
        bias_hidden_update = (1 / m) * np.sum(hidden_layer_error)

        return weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update

    def update_weights_and_biases(self, weights_input_hidden_update: np.ndarray, bias_hidden_update: np.ndarray, weights_hidden_output_update: np.ndarray, bias_output_update: np.ndarray) -> None:
        """
        Update the weights and biases of the neural network using the computed gradients.

        Args:
            weights_input_hidden_update (np.ndarray): Gradient of the weights between input and hidden layers.
            bias_hidden_update (np.ndarray): Gradient of the biases for the hidden layer.
            weights_hidden_output_update (np.ndarray): Gradient of the weights between hidden and output layers.
            bias_output_update (np.ndarray): Gradient of the bias for the output layer.
        """

        self.bias_output -= bias_output_update * self.learning_rate
        self.weights_hidden_output -= weights_hidden_output_update * self.learning_rate
        self.bias_hidden -= bias_hidden_update * self.learning_rate
        self.weights_input_hidden -= weights_input_hidden_update * self.learning_rate

    @staticmethod
    def get_prediction(output_layer_output: np.ndarray) -> np.ndarray:
        """
        Convert the output layer values to binary predictions (0 or 1) based on a threshold.

        Args:
            output_layer_output (np.ndarray): The output values from the output layer of the neural network.

        Returns:
            np.ndarray: The binary predictions (0 or 1) corresponding to the input output layer values.
        """
        return (output_layer_output > 0.5).astype(np.int32)

    @staticmethod
    def get_accuracy(true_values: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the predictions compared to the true values.

        Args:
            true_values (np.ndarray): The true values (ground truth) of the samples.
            predictions (np.ndarray): The predicted values for the samples.

        Returns:
            float: The accuracy of the predictions as a fraction of correct predictions over total predictions.
        """

        return np.sum(predictions == true_values) / true_values.size

    def gradient_descent(self, feature_sets: np.ndarray, label_sets: np.ndarray) -> float:
        """
        Perform gradient descent to optimize the weights and biases of the SpamClassifier.

        :param feature_sets: A list of numpy arrays containing the input features for each batch.
        :param label_sets: A list of numpy arrays containing the true labels for each batch.
        """
        for _ in range(100):
            for input_data, true_values in zip(feature_sets, label_sets):
                hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output = self.forward_propagation(
                    input_data)
                weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update = self.backward_propagation(
                    hidden_layer_input, hidden_layer_output, output_layer_output, input_data, true_values)
                self.update_weights_and_biases(
                    weights_input_hidden_update, bias_hidden_update, weights_hidden_output_update, bias_output_update)
                predictions = self.get_prediction(output_layer_output)
                accuracy = self.get_accuracy(true_values, predictions)
                print('Improving accuracy:', accuracy)

    def make_prediction(self, test_data: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given test data using the trained SpamClassifier.

        :param test_data: A numpy array containing the input features for the test data.
        :return: A numpy array containing the predicted labels for the test data.
        """
        _, _, _, output_layer_output = self.forward_propagation(test_data)
        test_predictions = self.get_prediction(output_layer_output)
        return test_predictions


# Load and process data
training_data, testing_data = load_data()
feature_sets, label_sets = create_batches(training_data)

# Initialize the classifier and train
classifier = SpamClassifier()
classifier.gradient_descent(feature_sets, label_sets)

# Test the classifier
test_features = testing_data[:, 1:]
test_labels = testing_data[:, 0].reshape(testing_data.shape[0], 1)
predictions = classifier.make_prediction(test_features)

# Calculate and display accuracy
accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]
print(f"Accuracy on test data is: {accuracy}")
