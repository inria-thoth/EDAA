import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftedReLU(nn.Module):
    def __init__(self, R: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(R))

    def forward(self, x):
        return F.relu(x - self.alpha)

def check_SReLU():
    x = torch.randn(16, 32)
    srelu = ShiftedReLU(32)
    y = srelu(x)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    check_SReLU()
