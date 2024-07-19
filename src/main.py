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




def train_nn(trainloader, testloader, model, criterion, optimizer, epochs, device='cpu'):
    """Train neural network model
    """

    network = model.to(device)

    print('Start training')

    loss_list = []
    acc_list = []

    for epoch in range(epochs):

        print('Epoch: {}'.format(epoch + 1))

        batch_loss = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs.to(device))
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        loss_list.append(np.mean(batch_loss))

        # Validate all classes
        acc_total = 0
        predictions_total = 0

        # Validate each class

        network.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                outputs = network(inputs.float().to(device))
                _, predictions = torch.max(outputs.data, 1)

                # sums over Boolean tensor, where element at index i is 1 if predictions[i]==labels[i]
                acc_batch = torch.sum(torch.eq(labels.to(device), predictions))
                predictions_total += predictions.size(dim=0)
                acc_total += acc_batch.item()

        # divide by total predictions
        acc_total /= predictions_total
        acc_list.append(acc_total)

        # revert model to training mode
        network.train()
        # Save the model

        torch.save(network.state_dict(), './mnist_epoch_{}_acc_{}.pth'.format(epoch + 1, acc_total))

    # plot loss and accuracy
    ep_range = np.arange(1, epochs + 1)

    return network, ep_range, loss_list, acc_list


def closure(model, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss


if __name__ == "__main__":
    # load data
    # IJCNN1
    dirData1 = "../data/ijcnn1/ijcnn1"
    testData1 = datahelper.load_ijcnn1(dirData1 + ".t")
    trainData1 = datahelper.load_ijcnn1(dirData1 + ".tr")
    # CoverType

    # MNIST
    mnist_trainloader, mnist_testloader = datahelper.load_mnist()

    # CNN with MNIST and SAGA
    cnn_model = models.CNN_simple()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    network, ep_range, loss_list, acc_list = train_nn(mnist_trainloader, mnist_testloader, cnn_model, criterion,
                                                      optimizer, epochs=10, device=device)
