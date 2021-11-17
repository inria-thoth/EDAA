import torch
import torch.nn as nn


class ASC_penalty(nn.Module):

    EPS = 1e-8

    def __init__(self, nu_init):
        super().__init__()
        self.nu = nn.Parameter(torch.ones(1), requires_grad=False)
        nn.init.constant_(self.nu, nu_init)

    def forward(self, alpha):
        nu = torch.clip(self.nu, self.EPS, 1.0)
        return (nu / 2) * ((alpha.sum(1) - 1) ** 2).mean(0)
