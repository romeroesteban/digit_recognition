import numpy as np
import pandas as pd
from tkinter import Tk
from DigitRecognizer import DigitRecognizer
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    root = Tk()

    data = pd.read_csv("binary_mnist.csv")

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:150].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[150:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    model = NeuralNetwork(20, 10)

    model.train(X_train, Y_train, 0.3, 1000)


    digit_recognizer = DigitRecognizer(root, model)
    root.mainloop()
