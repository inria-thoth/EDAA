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

class SAD(nn.Module):
  def __init__(self, num_bands: int=156):
    super(SAD, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    """Spectral Angle Distance Objective
    Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'
    
    Params:
        input -> Output of the autoencoder corresponding to subsampled input
                tensor shape: (batch_size, num_bands)
        target -> Subsampled input Hyperspectral image (batch_size, num_bands)
        
    Returns:
        angle: SAD between input and target
    """
    try:
      input_norm = torch.sqrt(torch.bmm(input.view(-1, 1, self.num_bands), input.view(-1, self.num_bands, 1)))
      target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))
      
      summation = torch.bmm(input.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
      angle = torch.acos(summation/(input_norm * target_norm))
      
    
    except ValueError:
      return 0.0
    
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
