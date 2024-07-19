import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogReg(nn.Module):
    """
    Logistic regression classifier
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Constructor for logistic regression classifier
        :param n_inputs: number of features in the input
        :param n_outputs: number of classes
        """
        super(LogReg, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        """
        forward pass for logistic regression classifier
        :param x: feature vector
        :return: prediction
        """
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class Perceptron():
    """Perceptron for binary classification"""
    ...


class SVM():
    """Support vector machine for multiclass classification"""


class CNN_simple(nn.Module):
    """Simple convolutional neural network for multiclass classification"""

    def __init__(self):
        super(CNN_simple, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        # flatten to (batch_size, ... )
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
