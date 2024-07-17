import torch
from torch.optim.optimizer import Optimizer, required


class SAGA(Optimizer):
    """Class implementing the SAGA optimizer for use in PyTorch, inherits from the class torch.optim.optimizer
       based on https://arxiv.org/pdf/1407.0202
    """

    def __init__(self, params, lr=1e-3):
        # Perform checks on parameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SAGA, self).__init__(params, defaults)

        # Initialize gradient history and average gradient
        self.grad_history = {}
        self.avg_grad = {}

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

                # Update average gradient
                self.avg_grad[p].add_(grad - self.grad_history[p])
                self.avg_grad[p].mul_(1 / (len(self.param_groups) * len(group['params'])))

                # SAGA update rule
                p.data.add_(-lr * (grad - self.grad_history[p] + self.avg_grad[p]))

                # Update gradient history
                self.grad_history[p].copy_(grad)

        return loss


class SVRG(Optimizer):
    """Class implementing the stochastic variance reduced gradient (SVRG) optimizer for use in PyTorch
       inherits from the class torch.optim.optimizer
       based on doi:10.5555/2999611.2999647
    """

    def __init__(self, params, lr=required):
        # Perform checks on parameters
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)

        # Store a snapshot of the parameters and the full gradient
        self.snapshot = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]
        self.full_grad = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, snapshot, full_grad in zip(self.param_groups, self.snapshot, self.full_grad):
            lr = group['lr']
            for p, s, g in zip(group['params'], snapshot, full_grad):
                if p.grad is None:
                    continue

                grad = p.grad.data

                # SVRG update rule
                p.data.add_(-lr * (grad - p.svrg_stored_grad + g))

        return loss

    def update_snapshot(self, model):
        """Update the snapshot and compute the full gradient."""
        for group, snapshot in zip(self.param_groups, self.snapshot):
            for p, s in zip(group['params'], snapshot):
                s.copy_(p.data)

        # Compute the full gradient and store it
        self.zero_grad()
        model.zero_grad()
        model_output = model.forward(model.input_data)
        model.loss(model_output, model.target_data).backward()

        for group, full_grad in zip(self.param_groups, self.full_grad):
            for p, g in zip(group['params'], full_grad):
                if p.grad is not None:
                    g.copy_(p.grad.data)

    def store_grad(self):
        """Store the current gradient for variance reduction."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if not hasattr(p, 'svrg_stored_grad'):
                        p.svrg_stored_grad = torch.zeros_like(p.grad.data)
                    p.svrg_stored_grad.copy_(p.grad.data)


class SAG(Optimizer):
    """Class implementing the stochastic average gradient (SAG) optimizer for use in PyTorch
       inherits from the class torch.optim.optimizer
       based on doi:10.1007/s10107-016-1030-6
    """

    def __init__(self, params, lr=1e-3):
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
