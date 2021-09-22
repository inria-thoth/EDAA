import os
import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import matplotlib.pyplot as plt
import numpy as np

from . import metrics


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseModel(pl.LightningModule):
    def __init__(
            self,
            optimizer=None,
            loss=None,
            save_figs_dir=None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.save_figs_dir = save_figs_dir
        self.loss = metrics.__dict__[loss]

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        # loss = F.mse_loss(x, x_hat)
        loss = self.loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        logger.debug("Configure optimizer")
        optimizer_class = torch.optim.__dict__[self.optimizer.class_name]
        optimizer = optimizer_class(self.parameters(), **self.optimizer.params)
        return optimizer

    def extract_abundances(self, x):
        raise NotImplementedError

    def extract_endmembers(self):
        raise NotImplementedError

    def plot_endmembers(self, save=True):
        endmembers = self.extract_endmembers()
        endmembers = endmembers.detach().numpy().T
        # Loop on the first dimension (R, B)
        fig, ax = plt.subplots(1,self.n_endmembers)
        for indx in range(self.n_endmembers):
            endmember = endmembers[indx]
            ax[indx].plot(endmember)
        if save:
            if not os.path.exists(self.save_figs_dir):
                os.makedirs(self.save_figs_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_figs_dir, "endmembers.png"))


    def plot_abundances(self, x, save=True):
        # Loop on the last dimensions to plot the abundances (H, W, R)
        abundances = self.extract_abundances(x)
        abundances = abundances.detach().numpy()
        fig, ax = plt.subplots(1,self.n_endmembers)
        for indx in range(self.n_endmembers):
            abund = abundances[:,:,indx]
            ax[indx].imshow(abund)
            ax[indx].get_xaxis().set_visible(False)
            ax[indx].get_yaxis().set_visible(False)
        if save:
            if not os.path.exists(self.save_figs_dir):
                os.makedirs(self.save_figs_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_figs_dir, "abundances.png"))
