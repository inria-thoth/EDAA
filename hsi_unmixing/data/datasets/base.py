import logging
import os
import pdb
import time

import hsi_unmixing.models.supervised as setters
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from hsi_unmixing import EPS
from hsi_unmixing.data import normalizers
from hsi_unmixing.data.noise_models import AdditiveWhiteGaussianNoise as AWGN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HSI:
    def __init__(
        self,
        name: str,
        data_path: str = "./data",
        normalizer="GlobalMinMax",
        setter="DecompSimplex",
    ):
        path = os.path.join(data_path, name)
        logger.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path)
        self.shortname = name.strip(".mat")

        data = sio.loadmat(path)
        logger.debug(f"Data keys: {data.keys()}")

        for key in filter(
            lambda s: not s.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(key, data[key])

        # self.dtype = np.float64
        # Data format check
        self.H = self.H.item()
        self.W = self.W.item()
        self.L = self.L.item()
        self.p = self.p.item()

        self.N = self.H * self.W

        # pdb.set_trace()
        assert self.E.shape == (self.L, self.p)
        assert self.Y.shape == (self.L, self.N)

        # Normalize Y
        Normalizer = normalizers.__dict__[normalizer]()
        self.Y = Normalizer.transform(self.Y)

        try:
            assert len(self.labels) == self.p
            # Curate labels from MATLAB string formatting
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]

        except AssertionError:
            # Create pseudo labels
            self.labels = list(np.arange(self.p))

        # Set GT abundances if not available
        try:
            self.__getattribute__("A")
        except AttributeError:
            self.set_GT_abundances(setter=setter)

        assert self.A.shape == (self.p, self.N)
        # Abundance Sum to One Constraint (ASC)
        assert np.allclose(self.A.sum(0), np.ones(self.N))
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint
        assert np.all(self.E >= -EPS)

        # Convert to Tensors
        self.Yt = torch.Tensor(self.Y)
        self.Et = torch.Tensor(self.E)
        self.At = torch.Tensor(self.A)

    def __repr__(self):
        msg = f"HSI => {self.shortname}\n"
        msg += "---------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples, ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({self.labels})\n"
        msg += f"GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n"
        return msg

    def set_GT_abundances(self, setter):
        """
        Compute GT abundances based on GT endmembers using decompSimplex
        """
        logger.info("Computing GT abundances...")

        tic = time.time()
        solver = setters.__dict__[setter]()
        self.A = solver.solve(self.Y, self.E)
        tac = time.time()

        logger.info(f"Computing GT abundances took {tac - tic:.2f}s")

    def __call__(self, asTensor=False):

        if asTensor:
            Y = self.Yt.clone()
            E = self.Et.clone()
            A = self.At.clone()

        else:
            Y = np.copy(self.Y)
            E = np.copy(self.E)
            A = np.copy(self.A)

        return (Y, E, A)

    def plot_endmembers(self, E0=None, normalize=False):
        """
        Display endmembers spectrum signature
        """
        title = f"{self.shortname}"
        if E0 is None:
            E = np.copy(self.E)
            title += " GT Endmembers"
        else:
            assert self.E.shape == E0.shape
            E = np.copy(E0)
            title += " Estimated Endmembers"
        if normalize:
            title += " ($l_\infty$-normalized)"
        plt.figure(figsize=(12, 4))
        for pp in range(self.p):
            data = E[:, pp]
            if normalize:
                data /= E[:, pp].max()
            plt.plot(data, label=self.labels[pp])
        plt.title(title)
        plt.legend(frameon=True)
        plt.show()

    def plot_abundances(self, A0=None, grid=None, transpose=False):
        """
        Display abundances maps
        """
        if grid is None:
            nrows, ncols = (1, self.p)
        else:
            assert len(grid) == 2
            nrows, ncols = grid
            assert nrows * ncols >= self.p

        title = f"{self.shortname}"
        if A0 is None:
            A = np.copy(self.A)
            title += " GT Abundances"
        else:
            assert self.A.shape == A0.shape
            A = np.copy(A0)
            title += " Estimated Abundances"
        A = A.reshape(self.p, self.H, self.W)
        if transpose:
            A = A.transpose(0, 2, 1)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 4 * nrows),
        )
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                curr_ax.imshow(A[kk, :, :])
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                kk += 1

                if kk == self.p:
                    break

        plt.suptitle(title)
        plt.show()

    def plot_hsi(
        self,
        Y0=None,
        SNR=None,
        channels=None,
        seed=0,
        sort_channels=True,
    ):

        if channels is None:
            # Generate 3 random channels
            generator = np.random.RandomState(seed=seed)
            channels = generator.randint(0, self.L - 1, size=3)
        assert len(channels) == 3
        if sort_channels:
            # Reorder the channels
            channels = np.sort(channels)

        if Y0 is None:
            Y = np.copy(self.Y)
        else:
            assert Y0.shape == (self.L, self.N)
            Y = np.copy(Y0)
        if SNR is not None:
            # Add noise
            Y = AWGN().fit_transform(Y, SNR, seed=seed)

        # Plot the image
        img = Y.reshape(self.L, self.H, self.W)
        img = img.transpose(1, 2, 0)

        colors = {key: value for (key, value) in zip("RGB", channels)}
        title = f"{self.shortname} Observation [SNR={SNR}dB]\n"
        title += "Colors: ("
        title += ", ".join([f"{k}:{v}" for (k, v) in colors.items()]) + ")\n"
        plt.title(title)
        plt.imshow(img[:, :, channels])
        plt.show()

    def plot_contributions(
        self,
        X0,
        grid=None,
        transpose=False,
        method=None,
    ):
        """
        Display pixels contribution maps
        """
        if grid is None:
            nrows, ncols = (1, self.p)
        else:
            assert len(grid) == 2
            nrows, ncols = grid
            assert nrows * ncols >= self.p

        title = f"{self.shortname} Pixels Contributions using {method}"
        assert self.A.shape == X0.shape
        X = np.copy(X0)
        X = X.reshape(self.p, self.H, self.W)
        if transpose:
            X = X.transpose(0, 2, 1)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 4 * nrows),
        )
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                curr_ax.imshow(X[kk, :, :], cmap="inferno")
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                kk += 1

                if kk == self.p:
                    break

        plt.suptitle(title)
        plt.show()


if __name__ == "__main__":
    hsi = HSI(
        "APEX4.mat",
        normalizer="GlobalMinMax",
        setter="DecompSimplex",
    )
    print(hsi)
    # hsi.plot_abundances()
    hsi.plot_contributions(hsi.A)
