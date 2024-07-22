import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


# Import of custom modules
import datahelper
import models
import optimizers
from losses import RegCrossEntropyLoss

device = ('cuda' if torch.cuda.is_available() else 'cpu')




def closure(model, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss


if __name__ == "__main__":
    np.random.seed(11)
    # load data
    # IJCNN1
    dirData1 = "../data/ijcnn1/ijcnn1"
    ijcnn1_testloader = datahelper.load_ijcnn1_to_dataloader(dirData1 + ".t", batch_size=32, shuffle=False)
    ijcnn1_trainloader = datahelper.load_ijcnn1_to_dataloader(dirData1 + ".tr", batch_size=32, shuffle=False)

    # MNIST
    mnist_trainloader, mnist_testloader = datahelper.load_mnist()

    # CNN with MNIST and SAGA
    cnn_model = models.CNN_simple()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=1e-3)
    # optimizer = optimizers.SAGA(cnn_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    network, ep_range, loss_list, acc_list, gradient_passes_list = optimizers.saga(mnist_trainloader, mnist_testloader,
                                                                                   cnn_model, optimizer=optimizer,
                                                                                   criterion=criterion, epochs=10,
                                                                                   device=device, prefix='mnist_saga')
