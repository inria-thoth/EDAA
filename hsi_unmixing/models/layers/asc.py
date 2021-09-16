import pdb
import logging

import torch
import torch.nn as nn

class ASC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Abundance Sum-to-one Constraint"""
        abundance = x / x.sum(0)
        return abundance

# TODO Check ASC
def check_asc():
    pass

if __name__ == "__main__":
    check_asc()
