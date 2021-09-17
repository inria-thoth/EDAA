# Generic class that holds models
import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseModel(pl.LightningModule):
    def __init__(
            self,
            dataset,
            save_figs_dir=None,
    ):
        super().__init__()

        self.save_figs_dir = save_figs_dir

        self.n_endmembers = dataset.n_endmembers
        self.n_bands = dataset.n_bands
        self.img_size = dataset.img_size

        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extract_abundances(self, x):
        raise NotImplementedError

    def extract_endmembers(self):
        raise NotImplementedError

    def plot_endmembers(self, save=True):
	endmembers = self.extract_endmembers()
        # TODO Loop on the first dimension (R, B)
	fig, ax = plt.subplots(1,self.n_endmembers)
	for indx in range(self.n_endmembers):
		endmember = endmembers[indx]
		ax[indx].plot(endmember)
	if save:
		plt.savefig(self.save_figs_dir + '/endmembers.png')
	plt.show()

    def plot_abundances(self, x, save=True):
        # TODO Loop on the last dimensions to plot the abundances (H, W, R)
	abundances = self.extract_abundances(x)
	fig, ax = plt.subplots(1,self.n_endmembers)
        for indx in range(self.n_endmembers):
		abund = abundaces[:,:,indx]
		ax[indx].imshow(abund)
		ax[indx].get_xaxis().set_visible(False)
    		ax[indx].get_yaxis().set_visible(False)
	if save:
		plt.savefig(self.save_figs_dir + '/abundances.png')
