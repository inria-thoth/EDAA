import pdb
import logging
import os

import scipy.io as sp
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .base import BaseDataset

class Samson(BaseDataset):

    n_bands = 156
    img_size = (95, 95)
    n_endmembers = 3

    img_folder = os.path.join("Samson", "Data_Matlab")
    gt_folder = os.path.join("Samson", "GroundTruth")

    img_fname = "samson_1.mat"
    gt_fname = "end3.mat"


    def __init__(self, path_data_dir):
        super().__init__(path_data_dir)

        training_data = sp.loadmat(self.path_img)
        # values 'V' shape => (B, H * W)
        labels = sp.loadmat(self.path_gt)
        # abundances 'A' shape => (R, H * W)
        # endmembers 'M' shape => (B, R)

        # reshape => (H * W, B)
        self.train_data = training_data["V"].T
        # reshape => (H * W, R)
        self.abundances = labels["A"].T
        # reshape => (R, B)
        self.endmembers = labels["M"].T


    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        return torch.Tensor(pixel)

    def __len__(self):
        return len(self.train_data)


def check_samson():
    from torch.utils.data import DataLoader

    batch_size = 16

    samson_dset = Samson("./data")
    train_dataloader = DataLoader(samson_dset,
                                  batch_size=batch_size,
                                  shuffle=True)
    x = next(iter(train_dataloader))
    assert x.shape[0] == batch_size
    assert x.shape[1] == samson_dset.n_bands


if __name__ == "__main__":
    check_samson()

