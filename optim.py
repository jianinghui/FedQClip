import torch


class QCSGD(torch.optim.Optimizer):
    def __init__(self, params, eta, gamma, device):
        super(QCSGD, self).__init__(params, {})
        self.eta = eta
        self.gamma = gamma
        self.lr = eta
        self.device = device

    def step(self, closure=False):

        gradient_norm = 0  # 梯度二范数和
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                gradient_norm += (torch.pow(param.grad.norm(2), 2)).item()
        gradient_norm = gradient_norm ** 0.5
        self.lr = min(self.eta, ((self.gamma * self.eta) / gradient_norm))
        # if self.lr !=self.eta:
        #     print(self.lr)
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                param.data -= self.lr * param.grad
        return gradient_norm
