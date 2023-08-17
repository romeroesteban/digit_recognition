import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self, hidden_size, output_size):
        self.W1 = np.random.rand(hidden_size, 784) - 0.5
        self.b1 = np.random.rand(hidden_size, 1) - 0.5
        self.W2 = np.random.rand(hidden_size, hidden_size) - 0.5
        self.b2 = np.random.rand(hidden_size, 1) - 0.5
        self.W3 = np.random.rand(output_size, hidden_size) - 0.5
        self.b3 = np.random.rand(output_size, 1) - 0.5

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def backward_prop(self, X, Y, Z1, A1, Z2, A2, Z3, A3):
        m, n = X.shape

        one_hot_Y = self.one_hot(Y)
        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = self.W3.T.dot(dZ3) * self.ReLU_deriv(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2, dW3, db3

    def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.W3 -= alpha * dW3
        self.b3 -= alpha * db3

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def ReLU_deriv(self, Z):
        return Z > 0

    def train(self, X_train, Y_train, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2, Z3, A3 = self.forward_prop(X_train)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(X_train, Y_train, Z1, A1, Z2, A2, Z3, A3)
            self.update_params(dW1, db1, dW2, db2, dW3, db3, alpha)

    def get_predictions(self, A3):
        return np.argmax(A3, axis=0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def predict(self, X):
        _, _, _, _, _, A3 = self.forward_prop(X)
        predictions = self.get_predictions(A3)
        return predictions
    
    def test_prediction(self, index, X, Y):
        current_image = X[:, index, None]
        prediction = self.predict(X[:, index, None])
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()
