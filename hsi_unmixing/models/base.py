# TODO generic class that holds models

import pdb
import logging

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class BaseModel(nn.Module):
    def __init__(self, dataset, save_figs_dir):
        super().__init__()

        self.save_figs_dir = save_figs_dir

        self.n_endmembers = dataset.n_endmembers
        self.bands = dataset.bands
        self.img_size = dataset.img_size

    def extract_abundances(self, x, plot=True):
        raise NotImplementedError

    def reconstruct(self, x):
        raise NotImplementedError

    def extract_endmembers(self):
        raise NotImplementedError

    def plot_endmembers(self, endmembers, save=True):
        # TODO Loop on the first dimension (R, B)
	fig, ax = plt.subplots(1,self.n_endmembers)
	for indx in range(self.n_endmembers):
		endmember = endmembers[indx]
		ax[indx].plot(endmember)
	if save:
		plt.savefig(self.save_figs_dir + '/endmembers.png')
	plt.show()

    def plot_abundances(self, abundances, save=True):
        # TODO Loop on the last dimensions to plot the abundances (H, W, R)
	fig, ax = plt.subplots(1,self.n_endmembers)
        for indx in range(self.n_endmembers):
		abund = abundaces[:,:,indx]
		ax[indx].imshow(abund)
		ax[indx].get_xaxis().set_visible(False)
    		ax[indx].get_yaxis().set_visible(False)
	if save:
		plt.savefig(self.save_figs_dir + '/abundances.png')
