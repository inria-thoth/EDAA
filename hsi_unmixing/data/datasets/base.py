import logging
import os
import time
import pdb

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import spams
import matplotlib.pyplot as plt

from hsi_unmixing.data.noise_models import AdditiveWhiteGaussianNoise as AWGN

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HSI:

    EPS = 1e-10

    def __init__(
        self,
        name: str,
        data_path: str = "./data",
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

        self.dtype = np.float64
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
        num = self.Y - self.Y.min()
        denom = self.EPS + (self.Y.max() - self.Y.min())
        self.Y = num / denom

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
            self.set_GT_abundances()

        assert self.A.shape == (self.p, self.N)
        # Abundance Sum to One Constraint (ASC)
        assert np.allclose(self.A.sum(0), np.ones(self.N))
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -self.EPS)
        # Endmembers Non-negative Constraint
        assert np.all(self.E >= -self.EPS)

    def __repr__(self):
        msg = f"HSI => {self.shortname}\n"
        msg += "---------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples, ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({self.labels})\n"
        msg += f"MinValue: {self.Y.min()}, MaxValue: {self.Y.max()}\n"
        return msg

    def set_GT_abundances(self):
        """
        Compute GT abundances based on GT endmembers using decompSimplex
        """
        Yf = np.asfortranarray(self.Y, dtype=self.dtype)
        Ef = np.asfortranarray(self.E, dtype=self.dtype)

        tic = time.time()
        W = spams.decompSimplex(Yf, Ef, computeXtX=True)
        tac = time.time()

        logging.info(f"Computing GT abundances took {tac - tic:.2f}s")

        self.A = sp.csr_matrix.toarray(W)

    def plot_endmembers(self, normalize=False):
        """
        Display endmembers spectrum signature
        """
        E = np.copy(self.E)
        title = f"{self.shortname} GT endmembers"
        if normalize:
            title += " ($l_\infty$-normalized)"
        plt.figure(figsize=(12, 4))
        for pp in range(self.p):
            # data = self.E[:, pp]
            data = E[:, pp]
            if normalize:
                data /= E[:, pp].max()
            plt.plot(data, label=self.labels[pp])
        plt.title(title)
        plt.legend(frameon=True)
        plt.show()

    def plot_abundances(self, grid=None, transpose=False):
        """
        Display abundances maps
        """
        if grid is None:
            nrows, ncols = (1, self.p)
        else:
            assert len(grid) == 2
            nrows, ncols = grid
            assert nrows * ncols >= self.p

        A = np.copy(self.A)
        A = A.reshape(self.p, self.H, self.W)
        if transpose:
            A = A.transpose(0, 2, 1)

        title = f"{self.shortname} GT abundances"
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

        Y = np.copy(self.Y)
        if SNR is not None:
            # Add noise
            Noise = AWGN(seed=seed)
            Y = Noise.fit_transform(Y, SNR)

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
