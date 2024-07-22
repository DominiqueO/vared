
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
from torch.utils.data import DataLoader


# TODO: Not fully functional, problems in step() function
class SAGAprototype(Optimizer):
    """
    Class implementing the SAGA optimizer for use in PyTorch
    inherits from the class torch.optim.Optimizer
    based on https://arxiv.org/pdf/1407.0202
    """

    def __init__(self, params, lr=1e-3, differentiable: bool=False):
        # Perform checks on parameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, differentiable=differentiable)
        super(SAGA, self).__init__(params, defaults)

        # Initialize gradient history and average gradient
        self.grad_history = {}
        self.avg_grad = {}

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.grad_history[p] = torch.zeros_like(p)
                    self.avg_grad[p] = torch.zeros_like(p)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            for p in params_with_grad:
                grad = p.grad


                # Update average gradient
                self.avg_grad[p].add_(grad - self.grad_history[p])
                # Update gradient history
                self.grad_history[p].copy_(grad)

                # SAGA update rule
                p.add_((grad - self.grad_history[p] + self.avg_grad[p]), alpha=-lr)

                # factor = 1 / (len(self.param_groups) * len(group['params']))




        return loss


def train_standard(trainloader, testloader, model, criterion, optimizer, epochs, lr=1e-3, device='cpu', prefix=''):
    """
    Normal procedure to train neural network model in PyTorch
    """

    network = model.to(device)

    print('Start training')

    loss_list = []
    acc_list = []
    # Store total number of gradient evaluations to compare optimization methods
    accumulated_gradient_passes = 0
    accumulated_gradient_passes_list = []

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
            accumulated_gradient_passes += 1
            optimizer.step()

        print("Loss: {}".format(np.mean(batch_loss)))
        loss_list.append(np.mean(batch_loss))
        accumulated_gradient_passes_list.append(accumulated_gradient_passes)

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
    torch.save(network.state_dict(), '../modelParams/{}_{}_acc_{}.pth'.format(prefix, epoch + 1, acc_total))

    # plot loss and accuracy
    ep_range = np.arange(1, epochs + 1)

    return network, ep_range, loss_list, acc_list, accumulated_gradient_passes_list


def saga(trainloader: DataLoader, testloader: DataLoader, model, optimizer, criterion, epochs: int, lr=1e-3,
         device='cpu', prefix=''):
    """
    Train neural network model with SAGA algorithm
    based on https://arxiv.org/pdf/1407.0202
    """

    network = model

    print('Start training')

    loss_list = []
    acc_list = []
    number_of_batches = len(trainloader)

    # Store total number of gradient evaluations to compare optimization methods
    accumulated_gradient_passes = 0
    accumulated_gradient_passes_list = []

    # Initialization: populate gradient tables and perform initial SGD step
    grad_history = []
    batch_loss = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero gradients
        network.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        # Compute average gradients over batch
        loss.backward()
        batch_loss.append(loss.item())
        # Because gradients have to be computed anyway, model is preoptimized
        optimizer.step()
        accumulated_gradient_passes += 1
        grads = get_gradients(network.parameters())
        grad_history.append(grads)

    accumulated_gradient_passes_list.append(accumulated_gradient_passes)
    loss_list.append(np.mean(batch_loss))

    # Compute average of batch tensors
    grad_average = get_param_averages(grad_history)


    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch + 1))
        # Draw random number
        ik = np.random.randint(low=0, high=number_of_batches)

        inputs, labels = get_batch_by_index(trainloader, ik)
        network.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        accumulated_gradient_passes += 1

        # SAGA update rule (update parameters and gradient average)
        saga_step(network.parameters(), grad_average, grad_history, ik, number_of_batches, lr=lr)
        # print(next(network.parameters()))

        print("Loss: {}".format(loss.item()))

        # Validate all classes
        acc_total = 0
        predictions_total = 0

        # Validate each class

        network.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                outputs = network(inputs.float())
                _, predictions = torch.max(outputs.data, 1)

                # sums over Boolean tensor, where element at index i is 1 if predictions[i]==labels[i]
                acc_batch = torch.sum(torch.eq(labels, predictions))
                predictions_total += predictions.size(dim=0)
                acc_total += acc_batch.item()

        # divide by total predictions
        acc_total /= predictions_total
        acc_list.append(acc_total)
        accumulated_gradient_passes_list.append(accumulated_gradient_passes)

        # revert model to training mode
        network.train()
        # Save the model

        # torch.save(network.state_dict(), '../modelParams/{}_epoch_{}_acc_{}.pth'.format(prefix, epoch + 1, acc_total))

    # Save the model
    torch.save(network.state_dict(), '../modelParams/{}_epoch_{}_acc_{}.pth'.format(prefix, epoch + 1, acc_total))

    # plot loss and accuracy
    ep_range = np.arange(1, epochs + 1)

    return network, ep_range, loss_list, acc_list, accumulated_gradient_passes_list


def get_gradients(params):
    grad_list = []
    for p in params:
        if p is not None:
            grad_list.append(p.grad)

    return grad_list


def get_param_averages(gradients_list):
    """
    Computes the averages of the gradients over a data set for each parameter
    :param gradients_list: list of list of parameter gradients,
    i.e. gradients_list[i][j] contains gradients of j-th parameter for i-th data point
    :return: list of averages of gradients over data points for each parameter
    """
    num_data_points = len(gradients_list)
    if num_data_points == 0:
        raise ValueError("The gradients_list is empty.")

    # Initialize a list of tensors to store the summed gradients
    summed_gradients = [torch.zeros_like(grad) for grad in gradients_list[0]]

    # Sum the gradients for each parameter
    for dp_gradients in gradients_list:
        for i, grad in enumerate(dp_gradients):
            summed_gradients[i] += grad

    # Calculate the average for each parameter
    avg_gradients = [summed_grad / num_data_points for summed_grad in summed_gradients]

    return avg_gradients




def saga_step(params, grad_average, grad_history, index, number_of_batches, lr=1e-3):
    """
    Performs one step according to the SAGA update rule
    updates the gradient history and the
    :param params: parameter of the model (e.g. model.parameters())
    :param grad_average: list of tensors containing the gradient averages
    :param grad_history: list of list of previously computed gradients
    :param index: index of sample data point
    :param number_of_batches: total number of batches
    :param lr: learning rate
    :return:
    """
    factor = 1 / number_of_batches
    with torch.no_grad():
        for k, p in enumerate(params, 0):
            if p.grad is not None:
                grad = p.grad
                old_grad = grad_history[index][k]
                grad_history[index][k].copy_(grad)
                # SAGA update
                p.add_(grad - old_grad + grad_average[k], alpha=-lr)
                # Update average gradient
                grad_average[k].add_(grad.mul_(factor) - old_grad.mul_(factor) )
    return


def get_batch_by_index(data_loader, batch_index):
    for idx, batch in enumerate(data_loader):
        if idx == batch_index:
            return batch
    raise IndexError("Batch index out of range")
