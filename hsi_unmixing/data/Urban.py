mport pdb
import logging
import os

import scipy.io as sp
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .base import BaseDataset

class Urban(BaseDataset):

    n_bands = 
    img_size =
    n_endmembers =

    img_folder = os.path.join()
    gt_folder = os.path.join()

    img_fname = 
    gt_fname = 


    def __init__(self, path_data_dir):
        super().__init__(path_data_dir)

        training_data = sp.loadmat(self.path_img)
        labels = sp.loadmat(self.path_gt)

        # reshape => (H * W, B)
        self.train_data = training_data
        # reshape => (H * W, R)
        self.abundances = labels
        # reshape => (R, B)
        self.endmembers = labels


    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        return torch.Tensor(pixel)

    def __len__(self):
        return len(self.train_data)


def check_urban():
    from torch.utils.data import DataLoader

    batch_size = 16

    urban_dset = Urban("./data")
    train_dataloader = DataLoader(urban_dset,
                                  batch_size=batch_size,
                                  shuffle=True)
    x = next(iter(train_dataloader))
    assert x.shape[0] == batch_size
    assert x.shape[1] == urban_dset.n_bands


if __name__ == "__main__":
    check_urban()

