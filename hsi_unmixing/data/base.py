import pdb
import logging
import os

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseDataset(Dataset):

    def __init__(
            self,
            path_data_dir,
    ):
        self.path_data_dir = path_data_dir
        self.path_imgs = os.path.join(path_data_dir, self.imgs)
        self.path_GT = os.path.join(path_data_dir, self.GT)

    def __getitem__(self, idx):
        # TODO implement this function
        pass

    def __len__(self):
        # TODO implement this function
        pass
