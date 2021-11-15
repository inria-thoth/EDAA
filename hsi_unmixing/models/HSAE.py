import logging
import pdb

import torch
import torch.nn as nn

from .base import BaseModel
from .layers import ASC, GaussianDropout, ShiftedReLU

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HSAE(BaseModel):
    def __init__(self, base, H, W, n_bands, n_endmembers, dropout=1.0, threshold=5, deep=True):
        super().__init__(**base)

        # Architecture sizes
        self.img_size = (H, W)
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers

        if deep:
            layers = [
                nn.Linear(self.n_bands, 9 * self.n_endmembers),
                nn.ReLU(),
                # nn.LeakyReLU(),
                nn.Linear(9 * self.n_endmembers, 6 * self.n_endmembers),
                nn.ReLU(),
                # nn.LeakyReLU(),
                nn.Linear(6 * self.n_endmembers, 3 * self.n_endmembers),
                nn.ReLU(),
                # nn.LeakyReLU(),
                nn.Linear(3 * self.n_endmembers, self.n_endmembers),
                nn.BatchNorm1d(self.n_endmembers),
                # nn.Softplus(threshold=threshold),
                ShiftedReLU(self.n_endmembers),
                ASC(),
                GaussianDropout(dropout),
            ]
        else:
            layers = [
                nn.Linear(self.n_bands, self.n_endmembers),
                nn.BatchNorm1d(self.n_endmembers),
                # nn.Softplus(threshold=threshold),
                ShiftedReLU(self.n_endmembers),
                ASC(),
                GaussianDropout(dropout),
            ]

        # Modules
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(self.n_endmembers, self.n_bands)

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

    def init_VCA(self, Y):
        """
        Y:  input image
            numpy.array => shape [L, N]
            L: number of channels
            N: number of pixels
        """
        # convert Y to numpy array
        # Y = Y.numpy()
        Ae, indice, Yp = VCA(Y, self.n_endmembers)
        # convert back to Tensor
        self.decoder.weight.data = torch.Tensor(Ae)

    def extract_endmembers(self):
        return self.decoder.weight.detach()

    def extract_abundances(self, x):
        self.eval()
        h = self.encoder(x)
        # Reshape the abundances => (H, W, R)
        abundances = h.reshape(*self.img_size, self.n_endmembers)
        return abundances


def check_HSAE():
    x = torch.randn(16, 156)

    cfg = {
        "H": 95,
        "W": 95,
        "n_endmembers": 3,
        "n_bands": 156,
        "dropout": 0.2,
        "threshold": 1.0,
        "deep": True,
        "base": {
            "loss": "sad",
            "save_figs_dir": "figures",
            "optimizer": {"class_name": "Adam", "params": {"lr": 0.001}},
        },
    }

    model = HSAE(**cfg)

    Y = torch.rand(156, 95 * 95)
    model.init_VCA(Y)
    model.eval()
    x_hat = model(x)
    assert x_hat.shape == x.shape


if __name__ == "__main__":
    check_HSAE()
