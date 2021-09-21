import pdb
import logging
import os

import torch
from torch.utils.data import Dataset
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseDataset(Dataset):

    def __init__(
            self,
            path_data_dir,
    ):
        self.path_data_dir = to_absolute_path(path_data_dir)

    def __getitem__(self, idx):
        pixel = self.train_data[idx]
        return torch.Tensor(pixel)

    def __len__(self):
        return len(self.train_data)
