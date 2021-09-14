# TODO Implement HSAE model

import pdb
import logging

import torch
import torch.nn as nn

from .base import BaseModel

class HSAE(BaseModel):
    def __init__(self, deep=False):
        super().__init__()
        self.deep = deep

        self.encoder = nn.Sequential()
        self.decoder = nn.Linear(self.n_endmembers, self.bands)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return encoded, reconstruction

    def extract_endmembers(self):
        return self.decoder.weight

    def reconstruct(self, x):
        _, reconstruction = self(x)
        return reconstruction

    def extract_abundances(self, x):
        encoded, _ = self(x)
        # TODO reshape the abundances => (H, W, R)
        return encoded
