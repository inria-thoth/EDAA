import torch.nn as nn


class SAD(nn.Module):
  def __init__(self, num_bands):
    super(SAD, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    """
	Spectral Angle Distance Objective
    """
    try:
      input_norm = torch.sqrt(torch.bmm(input.view(-1, 1, self.num_bands), input.view(-1, self.num_bands, 1)))
      target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))
      
      summation = torch.bmm(input.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
      angle = torch.acos(summation/(input_norm * target_norm))
      
    
    except ValueError:
      return 0.0
    
    return angle

class SID(nn.Module):
  def __init__(self, epsilon: float=1e5):
    super(SID, self).__init__()
    self.eps = epsilon

  def forward(self, input, target):
    """
	Spectral Information Divergence Objective
    """
    normalize_inp = (input/torch.sum(input, dim=0)) + self.eps
    normalize_tar = (target/torch.sum(target, dim=0)) + self.eps
    sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) + normalize_tar * torch.log(normalize_tar / normalize_inp))
    
    return sid
