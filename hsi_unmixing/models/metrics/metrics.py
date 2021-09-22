import os
import pdb
import logging

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sad(inputs: Tensor, targets: Tensor, reduction="mean") -> Tensor:
    assert inputs.shape == targets.shape
    _, B = inputs.shape

    inputs_norm = torch.norm(inputs, dim=1)
    targets_norm = torch.norm(targets, dim=1)

    summation = torch.bmm(inputs.view(-1, 1, B), targets.view(-1, B, 1))
    summation = summation.squeeze().squeeze()
    angle = torch.acos(summation / (inputs_norm * targets_norm))

    if reduction == "mean":
        return angle.mean(0)
    return angle


def mse(inputs: Tensor, targets: Tensor) -> Tensor:
    assert inputs.shape == targets.shape

    return F.mse_loss(inputs, targets)

def check_sad():
    inputs = torch.randn(16, 32)
    targets = torch.randn(16, 32)

    out = sad(inputs, targets)

    print(f"Out: {out}")
    print(f"Out shape: {out.shape}")

if __name__ == "__main__":
    check_sad()
