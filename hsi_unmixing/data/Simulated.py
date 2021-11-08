import pdb
import logging
import os

import scipy.io as sio

import torch

from .base import BaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseSimulated(BaseDataset):
    def __init__(self, path_data_dir):
        super().__init__(path_data_dir)

        self.img_size = (200, 200)
        self.n_bands = 224
        self.n_endmembers = 5

        img_path = os.path.join(
            self.path_data_dir,
            "Simulated",
            "Mixed_TrSet.mat",
        )
        abundances_path = os.path.join(
            self.path_data_dir,
            "Simulated",
            "TeLabel.mat",
        )

        # shape => (W, H, B)
        self.data = sio.loadmat(img_path)["Mixed_TrSet"]
        self.labels = sio.loadmat(abundances_path)["TeLabel"]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SimulatedPixels(BaseSimulated):
    def __init__(self, path_data_dir):
        super().__init__(path_data_dir)

        # cast as tensors
        X = torch.Tensor(self.data)
        self.Y = torch.Tensor(self.labels)

        # clamp data
        self.X = torch.clamp(X, 0.0, 1.0)

        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)


class SimulatedPatches(BaseSimulated):
    def __init__(self, path_data_dir):
        super().__init__(path_data_dir)

        # reshape data
        # shape => (W, H, B)
        # reshape data
        X = self.data.reshape(1, 200, 200, 224)
        Y = self.labels.reshape(1, 200, 200, 5)

        # cast as tensors
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)

        # clamp data
        X = torch.clamp(X, 0.0, 1.0)

        # permute axes
        # shape => (H, W, B)
        self.X = X.permute((0, 2, 1, 3))
        self.Y = Y.permute((0, 2, 1, 3))

        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)


def check_simulated_pixels():
    from torch.utils.data import DataLoader

    print("Testing simulated pixels...")
    batch_size = 32

    dset = SimulatedPixels("./data")
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True)

    batch = next(iter(dataloader))
    x, y = batch

    assert x.shape[1] == 224
    assert y.shape[1] == 5

    print("Tensors shapes as expected...")


def check_simulated_patches():
    from torch.utils.data import DataLoader

    print("Testing simulated patches...")

    dset = SimulatedPatches("./data")
    dataloader = DataLoader(dset, batch_size=1, shuffle=False)

    batch = next(iter(dataloader))
    x, y = batch

    assert x.shape[0] == 1
    assert x.shape[3] == 224
    assert y.shape[3] == 5

    print("Tensors shapes as expected...")


if __name__ == "__main__":
    check_simulated_pixels()
    check_simulated_patches()
