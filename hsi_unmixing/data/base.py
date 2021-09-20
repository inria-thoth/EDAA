import pdb
import logging
import os

from torch.utils.data import Dataset
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseDataset(Dataset):

    def __init__(
            self,
            path_data_dir,
            **kwargs,
    ):
        self.path_data_dir = to_absolute_path(path_data_dir)
        self.path_img = os.path.join(self.path_data_dir, self.img_folder, self.img_fname,)
        self.path_gt = os.path.join(self.path_data_dir, self.gt_folder, self.gt_fname,)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
