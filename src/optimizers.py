import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
import copy
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import collections


class SAG(Optimizer):
    """
    Class implementing the stochastic average gradient (SAG) optimizer for use in PyTorch
    inherits from the class torch.optim.optimizer
    based on doi:10.1007/s10107-016-1030-6
    """

    def __init__(self, params, lr=1e-3, differentiable: bool = False):
        # Perform check on parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(SAG, self).__init__(params, defaults)

        # Initialize gradient history and average gradient
        self.grad_history = {}
        self.avg_grad = {}
        self.num_samples = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.grad_history[p] = torch.zeros_like(p.data)
                    self.avg_grad[p] = torch.zeros_like(p.data)



    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if self.num_samples == 0:
                    # Initial population of grad_history and avg_grad
                    self.grad_history[p].copy_(grad)
                    self.avg_grad[p].copy_(grad)
                else:
                    # Update average gradient
                    self.avg_grad[p].add_(grad - self.grad_history[p])

                # SAG update rule
                p.data.add_(-lr * self.avg_grad[p])

                # Update gradient history
                self.grad_history[p].copy_(grad)

        # Increment the sample counter
        self.num_samples += 1

        return loss


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


class SVRG(Optimizer):
    """
    Class implementing the stochastic variance reduced gradient (SVRG) optimizer for use in PyTorch
    inherits from the class torch.optim.Optimizer
    based on doi:10.5555/2999611.2999647
    """

    def __init__(self, params, lr=0.01, epoch_size=None):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if epoch_size is not None and epoch_size <= 0:
            raise ValueError("Invalid epoch size: {}".format(epoch_size))

        defaults = dict(lr=lr, epoch_size=epoch_size)
        super(SVRG, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        if closure is None:
            raise ValueError("SVRG requires a closure to reevaluate the model and compute the loss")

        loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            epoch_size = group['epoch_size']
            params_with_grad = []
            grads = []
            states = []

            # Collect parameters and their gradients
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad.data)
                    states.append(copy.deepcopy(p.data))

            # Compute the full gradient
            if self._state['full_grad'] is None:
                self._state['full_grad'] = []
                for p in params_with_grad:
                    self._state['full_grad'].append(torch.zeros_like(p.data))

            full_grad = self._state['full_grad']
            for i, p in enumerate(params_with_grad):
                full_grad[i].zero_()
                full_grad[i].add_(p.grad.data)

            for i in range(epoch_size):
                def closure():
                    # Recompute the loss and gradient
                    loss = closure()
                    return loss

                # Compute the snapshot gradient
                snapshot_grads = []
                for p, g in zip(params_with_grad, grads):
                    snapshot_grads.append(copy.deepcopy(g))

                # Update parameters
                for j, p in enumerate(params_with_grad):
                    p.data = states[j] - lr * (snapshot_grads[j] - full_grad[j] + grads[j])

        return loss

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    # def __init__(self, params, lr=required):
    #     # Perform checks on parameters
    #     if lr is not required and lr < 0.0:
    #         raise ValueError("Invalid learning rate: {}".format(lr))
    #
    #     defaults = dict(lr=lr)
    #     super(SVRG, self).__init__(params, defaults)
    #
    #     # Store a snapshot of the parameters and the full gradient
    #     self.snapshot = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]
    #     self.full_grad = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]
    #
    # def step(self, closure=None):
    #     loss = None
    #     if closure is not None:
    #         loss = closure()
    #
    #     for group, snapshot, full_grad in zip(self.param_groups, self.snapshot, self.full_grad):
    #         lr = group['lr']
    #         for p, s, g in zip(group['params'], snapshot, full_grad):
    #             if p.grad is None:
    #                 continue
    #
    #             grad = p.grad.data
    #
    #             # SVRG update rule
    #             p.data.add_(-lr * (grad - p.svrg_stored_grad + g))
    #
    #     return loss
    #
    # def update_snapshot(self, model):
    #     """Update the snapshot and compute the full gradient."""
    #     for group, snapshot in zip(self.param_groups, self.snapshot):
    #         for p, s in zip(group['params'], snapshot):
    #             s.copy_(p.data)
    #
    #     # Compute the full gradient and store it
    #     self.zero_grad()
    #     model.zero_grad()
    #     model_output = model.forward(model.input_data)
    #     model.loss(model_output, model.target_data).backward()
    #
    #     for group, full_grad in zip(self.param_groups, self.full_grad):
    #         for p, g in zip(group['params'], full_grad):
    #             if p.grad is not None:
    #                 g.copy_(p.grad.data)
    #
    # def store_grad(self):
    #     """Store the current gradient for variance reduction."""
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             if p.grad is not None:
    #                 if not hasattr(p, 'svrg_stored_grad'):
    #                     p.svrg_stored_grad = torch.zeros_like(p.grad.data)
    #                 p.svrg_stored_grad.copy_(p.grad.data)


class SAGA2(Optimizer):
    def __init__(self, params, lr=required):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr)
        super(SAGA2, self).__init__(params, defaults)

        # Initialize gradient memory and parameter memory
        for group in self.param_groups:
            group['grad_table'] = []
            for p in group['params']:
                if p.requires_grad:
                    group['grad_table'].append(torch.zeros_like(p.data))
                else:
                    group['grad_table'].append(None)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            grad_table = group['grad_table']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data
                table_entry = grad_table[i]

                # Compute the average of stored gradients
                avg_grad = sum([table_entry[j] for j in range(len(table_entry))]) / len(table_entry)

                # Pick a random index
                j = torch.randint(0, len(table_entry), (1,)).item()

                # Compute the gradient difference
                grad_diff = grad - table_entry[j]

                # Update the gradient table entry
                # table_entry[j] = grad

                # Update the parameter
                p.data = p.data - lr * (grad_diff + avg_grad)

        return loss


def train_standard(trainloader, testloader, model, criterion, optimizer, epochs, device='cpu'):
    """
    Train neural network model
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

        torch.save(network.state_dict(), '../modelParams/mnist_epoch_{}_acc_{}.pth'.format(epoch + 1, acc_total))

    # plot loss and accuracy
    ep_range = np.arange(1, epochs + 1)

    return network, ep_range, loss_list, acc_list


def saga(trainloader: DataLoader, testloader: DataLoader, model, optimizer, criterion, epochs: int, lr=1e-3,
         device='cpu', prefix=''):
    """
    Train neural network model with SAGA algorithm
    """

    network = model.to(device)

    print('Start training')

    loss_list = []
    acc_list = []
    number_of_batches = len(trainloader)

    # Store total number of gradient evaluations to compare optimization methods
    accumulated_gradient_passes = 0
    accumulated_gradient_passes_list = []

    # Initialization: populate gradient tables and perform initial SGD step
    grad_history = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero gradients
        optimizer.zero_grad()
        outputs = network(inputs.to(device))
        loss = criterion(outputs, labels)
        # Compute average gradients over batch
        loss.backward()
        # Because gradients have to be computed anyway, model is preoptimized
        optimizer.step()
        accumulated_gradient_passes += 1
        grads = get_gradients(network.parameters())
        grad_history.append(grads)

    accumulated_gradient_passes_list.append(accumulated_gradient_passes)

    # Compute average of batch tensors
    grad_average = get_param_averages(grad_history)


    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch + 1))
        # Draw random number
        ik = np.random.randint(low=0, high=number_of_batches)

        inputs, labels = get_batch_by_index(trainloader, ik)
        optimizer.zero_grad()
        outputs = network(inputs.to(device))
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        print("Loss: {}".format(loss.item()))
        loss.backward()
        accumulated_gradient_passes += 1

        # SAGA update rule (update parameters and gradient average)
        saga_step(model.parameters(), grad_average, grad_history, ik, lr=lr)

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
        accumulated_gradient_passes_list.append(accumulated_gradient_passes)

        # revert model to training mode
        network.train()
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




def saga_step(params, grad_average, grad_history, index, lr=1e-3):
    grad_list = []
    for k, p in enumerate(params, 0):
        if p is not None:
            grad = p.grad
            old_grad = grad_history[index][k]
            grad_history[index][k].copy_(grad)
            # SAGA update
            p.detach().add_(grad - old_grad + grad_average[k], alpha=-lr)
            # Update average gradient
            grad_average[k].add_(grad - old_grad)
    return

def get_batch_by_index(data_loader, batch_index):
    for idx, batch in enumerate(data_loader):
        if idx == batch_index:
            return batch
    raise IndexError("Batch index out of range")
