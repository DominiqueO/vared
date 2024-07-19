
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
    def __init__(self, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 60, 3, padding = 'same', bias = True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 40, 3, padding = 'same', bias = True)
        self.fc1 = nn.Linear(40 * (image_size//4)**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, image_size**2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(
