import pdb
import logging
import os

import scipy.io as sp
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .base import BaseDataset

class Cuprite(BaseDataset):

    img_size = (250, 190)
    n_endmembers = 12

    img_folder = os.path.join("Cuprite","Data_Matlab")
    gt_folder = os.path.join("Cuprite","groundTruth_Cuprite_end12")

    gt_fname = "groundTruth_Cuprite_nEnd12.mat"


    def __init__(self, path_data_dir, n_bands=188):
        super().__init__(path_data_dir)

        if n_bands == 188:
            self.n_bands = 188
            self.img_fname = "CupriteS1_R188.mat"
        else: 
            self.n_bands = 224
            self.img_fname = "CupriteS1_F224.mat"


        self.path_img = os.path.join(self.path_data_dir, self.img_folder, self.img_fname)
        self.path_gt = os.path.join(self.path_data_dir, self.gt_folder, self.gt_fname)

        training_data = sp.loadmat(self.path_img)
        labels = sp.loadmat(self.path_gt)

        pdb.set_trace()

        # reshape => (H * W, B)
        self.train_data = training_data['Y'].T
        # reshape => (H * W, R)
        self.abundances = None ## This dataset doesnt has Abundaces
        # reshape => (R, B)
        self.endmembers = labels['M'].T

    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        # TODO convert this pixel to a fitting tensor type
        return torch.Tensor(pixel.astype('float32'))

def check_cuprite():
    from torch.utils.data import DataLoader

    batch_size = 16

    cuprite_dset = Cuprite("./data", n_bands=188)
    train_dataloader = DataLoader(cuprite_dset,
                                  batch_size=batch_size,
                                  shuffle=True)
    x = next(iter(train_dataloader))
    assert x.shape[0] == batch_size
    assert x.shape[1] == cuprite_dset.n_bands


if __name__ == "__main__":
    check_cuprite()

