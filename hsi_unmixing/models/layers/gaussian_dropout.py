import pdb
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()

        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sampling e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.training:
            # Sampling
            epsilon = torch.randn(x.size()) * self.alpha + 1
            # Turn into a torch Variable
            epsilon = Variable(epsilon)
            # Transfer to CUDA if available
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        return x


def check_GD(alpha: float = 1.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gd = GaussianDropout(alpha=alpha)

    inp = torch.randn(95*95, 156).to(device)

    out = gd(inp)


if __name__ == "__main__":
    check_GD()
