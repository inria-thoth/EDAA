import pdb
import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from .base import BaseModel
from .losses import ASC_penalty


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SparseCoding_pw(pl.LightningModule):

    EPS = 1e-12

    def __init__(
        self,
        n_bands,
        n_endmembers,
        unrollings,
        lambd_init=0.1,
        use_C=False,
        use_W=False,
    ):

        super().__init__()
        # Define D, eta, lambda
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.unrollings = unrollings
        self.use_C = use_C
        self.use_W = use_W

        weight_init = self._init_dictionary()
        self.D = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
        self.D.weight.data = weight_init

        self.eta = nn.Parameter(torch.zeros(1))
        self.lambd = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.eta, 1.0)
        nn.init.constant_(self.lambd, lambd_init)

        if self.use_C:
            self.C = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
            self.C.weight.data = weight_init
        if self.use_W:
            self.W = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
            self.W.weight.data = weight_init

    # special routine to initialize D (needs to be positive)
    def _init_dictionary(self):
        D = torch.rand(size=(self.n_bands, self.n_endmembers))
        dtd = D.t() @ D
        e, _ = torch.symeig(dtd, eigenvectors=False)
        D /= torch.sqrt(torch.max(e))
        return D

    def forward(self, x):
        # Return sparse code and reconstruction

        # Compute g and the first sparse code iterate
        # (b, R)
        if self.use_C:
            g = self.eta * F.linear(x, self.C.weight.t())
        else:
            g = self.eta * F.linear(x, self.D.weight.t())
        alpha = F.relu(g - self.lambd)

        # G trick
        if self.unrollings > 1:
            # retrieve the weight => (L, R)
            D = self.D.weight
            # build - eta * D.T() @ D => (R, R)
            if self.use_C:
                C = self.C.weight
                G = self.eta * torch.neg(C.t() @ D)
            else:
                G = self.eta * torch.neg(D.t() @ D)
            G.diagonal().add_(1.0)  # add identity => (R, R)

        # Unfoldings
        for ii in range(self.unrollings - 1):
            pre_alpha = g + F.linear(alpha, G)
            alpha = F.relu(pre_alpha - self.lambd)

        # l1 normalization
        # alpha => (b, R)
        alpha = alpha / (alpha.sum(1, keepdims=True) + self.EPS)

        # Compute reconstruction
        # recon => (b, L)
        if self.use_W:
            recon = F.linear(alpha, self.W.weight)
        else:
            recon = F.linear(alpha, self.D.weight)
        return recon, alpha

    def training_step(self, batch, batch_idx):
        y, a = batch
        y_hat, code = self(y)
        loss = F.mse_loss(y, y_hat) + F.mse_loss(a, code)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def extract_endmembers(self):
        return self.W.weight.data

    def extract_abundances(self, x):
        with torch.no_grad():
            _, code = self(x)
        return code.view(95, 95, -1)

    def plot_abundances(self, x, save=True):
        # Loop on the last dimensions to plot the abundances (H, W, R)
        abundances = self.extract_abundances(x)
        abundances = abundances.detach().numpy()
        fig, ax = plt.subplots(1, self.n_endmembers)
        for indx in range(self.n_endmembers):
            abund = abundances[:, :, indx]
            ax[indx].imshow(abund)
            ax[indx].get_xaxis().set_visible(False)
            ax[indx].get_yaxis().set_visible(False)


class SC_ASC_pw(nn.Module):

    EPS = 1e-12

    def __init__(
        self,
        n_bands,
        n_endmembers,
        unrollings,
        # lambd_init=0.1,
        nu_init=0.1,
        gamma_init=0.1,
        use_C=False,
        use_W=False,
    ):

        super().__init__()
        # Define D, eta, lambda
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.unrollings = unrollings
        self.use_C = use_C
        self.use_W = use_W

        weight_init = self._init_dictionary()
        self.D = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
        self.D.weight.data = weight_init

        self.eta = nn.Parameter(torch.zeros(1))
        # self.lambd = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.eta, 1.0)
        # nn.init.constant_(self.lambd, lambd_init)
        nn.init.constant_(self.gamma, gamma_init)
        self.ASC = ASC_penalty(nu_init)

        if self.use_C:
            self.C = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
            self.C.weight.data = weight_init
        if self.use_W:
            self.W = nn.Linear(self.n_bands, self.n_endmembers, bias=False)
            self.W.weight.data = weight_init

    # special routine to initialize D (needs to be positive)
    def _init_dictionary(self):
        D = torch.rand(size=(self.n_bands, self.n_endmembers))
        dtd = D.t() @ D
        e, _ = torch.symeig(dtd, eigenvectors=False)
        D /= torch.sqrt(torch.max(e))
        return D

    def forward(self, x):
        # Return sparse code and reconstruction

        # Compute g and the first sparse code iterate
        # (b, R)
        if self.use_C:
            g = F.linear(x, self.C.weight.t())
        else:
            g = self.eta * F.linear(x, self.D.weight.t())
        alpha = F.relu(g + self.eta * self.gamma * torch.ones_like(g))
        # (b, )
        lambd = self.gamma * (alpha.sum(1) - 1)

        # G trick
        if self.unrollings > 1:
            # retrieve the weight => (L, R)
            D = self.D.weight
            # build - eta * D.T() @ D => (R, R)
            if self.use_C:
                C = self.C.weight
                G = torch.neg(C.t() @ D)
            else:
                G = self.eta * torch.neg(D.t() @ D)
            G.diagonal().add_(1.0)  # add identity => (R, R)

        # Unfoldings
        for ii in range(self.unrollings - 1):
            pre_alpha = g + F.linear(alpha, G)
            penalty = (lambd + self.gamma * (alpha.sum(1) - 1)) * torch.ones_like(g)
            alpha = F.relu(pre_alpha + penalty)
            lambd = lambd + self.gamma * (alpha.sum(1) - 1)

        # Compute reconstruction
        # recon => (b, L)
        if self.use_W:
            recon = F.linear(alpha, self.W.weight)
        else:
            recon = F.linear(alpha, self.D.weight)
        return recon, alpha


def check_SC():
    B, L, R, K = 10, 31, 5, 12
    y = torch.randn(B, L)
    model = SparseCoding_pw(L, R, K)
    y_hat, alpha = model(y)
    assert y_hat.shape == (B, L)
    assert alpha.shape == (B, R)


if __name__ == "__main__":
    check_SC()
