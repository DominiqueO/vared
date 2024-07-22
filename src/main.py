import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


# Import of custom modules
import datahelper
import models
import optimizers
from losses import RegCrossEntropyLoss

matplotlib.use('macosx')
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
    mnist_trainloader, mnist_testloader = datahelper.load_mnist(batch_size=32, shuffle=False)

    # CNN with MNIST and SAGA
    cnn_model = models.CNN_simple()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.1)
    # optimizer = optimizers.SAGA(cnn_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()


    network, ep_range, loss_list, acc_list, gradient_passes_list = optimizers.saga(mnist_trainloader, mnist_testloader,
                                                                                   cnn_model, optimizer=optimizer,
                                                                                   criterion=criterion, epochs=100, lr=0.01,
                                                                                   device=device, prefix='mnist_saga')

    network_sgd, ep_range_sgd, loss_list_sgd, acc_list_sgd, gradient_passes_list_sgd = optimizers.train_standard(mnist_trainloader, mnist_testloader,
                                                                                   cnn_model, optimizer=optimizer,
                                                                                   criterion=criterion, epochs=10, lr=0.01,
                                                                                   device=device, prefix='mnist_sgd')

    # Baseline / reference model
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    network_ref, ep_range_ref, loss_list_ref, acc_list_ref, gradient_passes_list_ref = optimizers.train_standard(
        mnist_trainloader, mnist_testloader,
        cnn_model, optimizer=optimizer,
        criterion=criterion, epochs=10, lr=0.001,
        device=device, prefix='mnist_adam')


    # TODO: move plots to separate method to improve code maintainability
    # Plot losses
    f_loss = plt.figure()
    plt.title("MNIST: Loss")
    plt.plot(gradient_passes_list_sgd, loss_list_sgd, label="SGD", linestyle="-")
    plt.plot(gradient_passes_list, loss_list, label="SAGA",  linestyle="-")
    plt.plot(gradient_passes_list_ref, loss_list_ref, label="reference", linestyle="-")
    plt.legend()
    plt.xlabel("gradient passes")
    plt.ylabel("loss")
    plt.show()
    plt.savefig("mnist_cnn_losses.pdf", format="pdf")

    # Plot accuracy
    f_acc = plt.figure()
    plt.title("MNIST: Accuracy")
    plt.plot(gradient_passes_list_sgd, acc_list_sgd, label="SGD", linestyle="-")
    plt.plot(gradient_passes_list[1:], acc_list, label="SAGA",  linestyle="-")
    plt.plot(gradient_passes_list_ref, acc_list_ref, label="reference", linestyle="-")
    plt.legend()
    plt.xlabel("gradient passes")
    plt.ylabel("accuracy")
    plt.show()
    plt.savefig("mnist_cnn_accuracy.pdf", format="pdf")