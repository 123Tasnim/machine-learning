# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

n_samples = 100
n_features = 8
n_classes = 4

# Generate dataset of 100 samples, 8 input features and 4 output classes
X, y = make_classification(
    n_samples=n_samples,  # row number
    n_features=n_features,  # feature numbers
    n_informative=5,  # feature numbers
    n_classes=n_classes,  # The number of classes
    random_state=42  # random seed
)

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

# One-hot encode training output
onehot_encoder = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(len(y_train), 1)
y_train_encoded = onehot_encoder.fit_transform(y_train)

# Print training data
for i in range(10):
    print('Training data features and output:')
    print(X_train[i])
    print(y_train_encoded[i])


class NeuralNetwork(object):
    def __init__(self):
        inputLayerNeurons = 8
        hiddenLayer1Neurons = 10
        hiddenLayer2Neurons = 10
        outLayerNeurons = 4

        self.learning_rate = 0.2

        # Initialize weight matrices for hidden and output layers
        self.W_HI_1 = np.random.randn(inputLayerNeurons, hiddenLayer1Neurons)
        self.W_HI_2 = np.random.randn(hiddenLayer1Neurons, hiddenLayer2Neurons)
        self.W_HO = np.random.randn(hiddenLayer2Neurons, outLayerNeurons)

    def sigmoid(self, x, der=False):
        if der:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feedForward(self, X):
        # Feedforward propagation
        self.hidden_input_1 = np.dot(X, self.W_HI_1)
        self.hidden_output_1 = self.sigmoid(self.hidden_input_1)

        self.hidden_input_2 = np.dot(self.hidden_output_1, self.W_HI_2)
        self.hidden_output_2 = self.sigmoid(self.hidden_input_2)

        self.output_input = np.dot(self.hidden_output_2, self.W_HO)
        self.output = self.sigmoid(self.output_input)

        return self.output

    def backPropagation(self, X, Y, pred):
        # Backpropagation steps
        output_error = Y - pred
        output_delta = output_error * self.sigmoid(pred, der=True)

        hidden_error_2 = output_delta.dot(self.W_HO.T)
        hidden_delta_2 = hidden_error_2 * self.sigmoid(self.hidden_output_2, der=True)

        hidden_error_1 = hidden_delta_2.dot(self.W_HI_2.T)
        hidden_delta_1 = hidden_error_1 * self.sigmoid(self.hidden_output_1, der=True)

        # Update weights
        self.W_HO += self.hidden_output_2.T.dot(output_delta) * self.learning_rate
        self.W_HI_2 += self.hidden_output_1.T.dot(hidden_delta_2) * self.learning_rate
        self.W_HI_1 += X.T.dot(hidden_delta_1) * self.learning_rate

    def train(self, X, Y):
        output = self.feedForward(X)
        self.backPropagation(X, Y, output)


# Create and train the neural network
NN = NeuralNetwork()
err = []
for i in range(5000):
    NN.train(X_train, y_train_encoded)
    err.append(np.mean(np.square(y_train_encoded - NN.feedForward(X_train))))

# Plot the training error
print("Error on training data:")
plt.plot(err)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Training Error')
plt.show()

# Run the trained model on test data
y_pred = NN.feedForward(X_test)
print("Model output:")
print(y_pred)

# One-hot encoded predictions
new_y_pred = np.zeros(y_pred.shape)
max_y_pred = np.argmax(y_pred, axis=1)
for i in range(len(y_pred)):
    new_y_pred[i][max_y_pred[i]] = 1

print("One-hot encoded output:")
print(new_y_pred)

# Obtain predicted output values
y_pred = new_y_pred.argmax(axis=1)
print("Predicted values: ", y_pred)

# Print true output values
y_test = y_test.flatten()
print("Actual values: ", y_test)


# Obtain accuracy on test data
def accuracy(y_pred, y_true):
    acc = y_pred == y_true
    print("Predictions: ", acc)
    return acc.mean()


print("Accuracy: ", accuracy(y_pred, y_test) * 100, "%")

# Print confusion matrix
confusion_matrix = metrics.confusion_matrix(np.array(y_test), np.array((y_pred)))
print("Confusion matrix: \n", confusion_matrix)
