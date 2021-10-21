import pdb
import logging
import os

import scipy.io as sp
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .base import BaseDataset

class JasperRidge(BaseDataset):

    img_size = (100, 100)
    n_endmembers = 4
    n_bands = 198

    img_folder = os.path.join("Jasper_Ridge","Data_Matlab")
    gt_folder = os.path.join("Jasper_Ridge","GroundTruth")

    gt_fname = "end4.mat"


    def __init__(self, path_data_dir, H=100, W=100, n_endmembers=4, n_bands=198):
        super().__init__(path_data_dir)

        # Assertions
        assert self.img_size == (H, W)
        assert self.n_bands == n_bands
        assert self.n_endmembers == n_endmembers

        if n_bands == 198:
            self.n_bands = 198
            self.img_fname = "jasperRidge2_R198.mat"
        else: 
            self.n_bands = 224
            self.img_fname = "jasperRidge2_F224_2.mat"

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
        self.set_labels(labels["cood"])
        logger.info(f"Label mapping: {self.labels}")

    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        return torch.Tensor(pixel.astype('float32'))

def check_jasper_ridge():
    from torch.utils.data import DataLoader

    batch_size = 16

    jasper_ridge_dset = JasperRidge("./data")
    train_dataloader = DataLoader(jasper_ridge_dset,
                                  batch_size=batch_size,
                                  shuffle=True)
    x = next(iter(train_dataloader))
    assert x.shape[0] == batch_size
    assert x.shape[1] == jasper_ridge_dset.n_bands


if __name__ == "__main__":
    check_jasper_ridge()

