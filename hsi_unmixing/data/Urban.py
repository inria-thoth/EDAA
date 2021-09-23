import pdb
import logging
import os

import scipy.io as sp
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .base import BaseDataset

class Urban(BaseDataset):

    img_size = (307,307)
    n_bands = 162
    n_endmembers = 4

    img_folder = os.path.join("Urban","Data_Matlab")
    gt_folder = os.path.join("Urban","GroundTruth")


    def __init__(self, path_data_dir, H=307, W=307, n_endmembers=4, n_bands=162 ):

        super().__init__(path_data_dir)
        if n_bands == 162:
            self.n_bands = 162
            self.img_fname = "Urban_R162.mat"
        else: 
            self.n_bands = 210
            self.img_fname = "Urban_F210.mat"
        
        if n_endmembers==4: 
            self.n_endmembers = 4
            self.gt_fname = os.path.join("groundTruth_Urban_end4","end4_groundTruth.mat")
        elif n_endmembers ==5: 
            self.n_endmembers = 5
            self.gt_fname = os.path.join("groundTruth_Urban_end5","end5_groundTruth.mat")
        else: 
            self.n_endmembers = 6
            self.gt_fname = os.path.join("groundTruth_Urban_end6","end6_groundTruth.mat")

        self.path_img = os.path.join(self.path_data_dir, self.img_folder, self.img_fname)
        self.path_gt = os.path.join(self.path_data_dir, self.gt_folder, self.gt_fname)

        training_data = sp.loadmat(self.path_img)
        labels = sp.loadmat(self.path_gt)

        # reshape => (H * W, B)
        self.train_data = training_data['Y'].T
        # reshape => (H * W, R)
        self.abundances = labels['A'].T
        # reshape => (R, B)
        self.endmembers = labels['M'].T

    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        return torch.Tensor(pixel.astype('float32'))


def check_urban():
    from torch.utils.data import DataLoader

    batch_size = 16

    urban_dset = Urban("./data", n_bands=162, n_endmembers=4)
    train_dataloader = DataLoader(urban_dset,
                                  batch_size=batch_size,
                                  shuffle=True)
    x = next(iter(train_dataloader))
    assert x.shape[0] == batch_size
    assert x.shape[1] == urban_dset.n_bands


if __name__ == "__main__":
    check_urban()

