# TODO Implement HSAE model

import pdb
import logging

import torch
import torch.nn as nn

from .base import BaseModel
from .layers import ASC, GaussianDropout


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HSAE(BaseModel):
    def __init__(self, dropout=1.0, threshold=5, deep=True):
        super().__init__()

        # Layers
        self.asc = ASC()
        self.gd = GaussianDropout(dropout)

        if deep:
            layers = [
                nn.Linear(self.n_bands, 9 * self.n_endmembers),
                nn.ReLU(),
                nn.Linear(9 * self.n_endmembers, 6 * self.n_endmembers),
                nn.ReLU(),
                nn.Linear(6 * self.n_endmembers, 3 * self.n_endmembers),
                nn.ReLU(),
                nn.Linear(3 * self.n_endmembers, self.n_endmembers),
                nn.BatchNorm1d(self.n_endmembers),
                nn.Softplus(threshold=threshold),
                self.asc,
                self.gd,
            ]
        else:
            layers = [
                nn.Linear(self.n_bands, self.n_endmembers),
                nn.BatchNorm1d(self.n_endmembers),
                nn.Softplus(threshold=threshold),
                self.asc,
                self.gd,

            ]

        # Modules
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(self.n_endmembers, self.n_bands)

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

    def extract_endmembers(self):
        return self.decoder.weight

    def extract_abundances(self, x):
        self.eval()
        h = self.encoder(x)
        # Reshape the abundances => (H, W, R)
        abundances = h.reshape(*self.img_size, self.bands)
        return abundances
