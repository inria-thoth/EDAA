import torch
import torch.nn as nn

# def ASC_penalty(alpha, nu):
#     return (nu / 2) * ((alpha.sum(1) - 1) ** 2).mean(0)


class ASC_penalty(nn.Module):
    def __init__(self, nu_init):
        super().__init__()
        self.nu = nn.Parameter(torch.ones(1))
        nn.init.constant_(self.nu, nu_init)

    def forward(self, alpha):
        return (self.nu / 2) * ((alpha.sum(1) - 1) ** 2).mean(0)
