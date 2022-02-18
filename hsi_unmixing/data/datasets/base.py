import logging
import os
import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import torch
from hsi_unmixing import EPS
from hsi_unmixing.data.noises import AdditiveWhiteGaussianNoise as AWGN
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HSI:
    def __init__(
        self,
        name: str,
        data_dir: str = "./data",
        figs_dir: str = "./figs",
        normalizer=None,
        setter=None,
    ):
        path = to_absolute_path(os.path.join(data_dir, name))
        logger.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path)
        self.shortname = name.strip(".mat")

        data = sio.loadmat(path)
        logger.debug(f"Data keys: {data.keys()}")

        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(key, data[key])

        # Data format check
        self.H = self.H.item()
        self.W = self.W.item()
        self.L = self.L.item()
        self.p = self.p.item()

        self.N = self.H * self.W

        assert self.E.shape == (self.L, self.p)
        assert self.Y.shape == (self.L, self.N)

        # Normalize Y
        # Normalizer = normalizers.__dict__[normalizer]()
        if normalizer is not None:
            self.Y = normalizer.transform(self.Y)
            self.scaledE = normalizer.transform(self.E)

        try:
            assert len(self.labels) == self.p
            # Curate labels from MATLAB string formatting
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]

        except AssertionError:
            # Create pseudo labels
            # self.labels = list(np.arange(self.p))
            self.labels = [f"#{ii}" for ii in range(self.p)]

        # Set GT abundances if not available
        try:
            self.__getattribute__("A")
        except AttributeError:
            if setter is not None:
                self.set_GT_abundances(setter=setter)
            else:
                raise ValueError(f"setter cannot be {setter}")

        assert self.A.shape == (self.p, self.N)
        # Abundance Sum to One Constraint (ASC)
        assert np.allclose(self.A.sum(0), np.ones(self.N))
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint
        assert np.all(self.E >= -EPS)

        # Save figures path
        self.figs_dir = figs_dir
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

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
        Compute GT abundances based on GT endmembers using a specific setter
        """
        logger.info("Computing GT abundances...")
        assert hasattr(setter, "solve")

        tic = time.time()
        self.A = setter.solve(self.Y, self.E)
        tac = time.time()

        logger.info(f"Computing GT abundances took {tac - tic:.2f}s")

    def __call__(self, asTensor=False):

        if asTensor:
            Y = torch.Tensor(self.Y)
            E = torch.Tensor(self.E)
            A = torch.Tensor(self.A)

        else:
            Y = np.copy(self.Y)
            E = np.copy(self.E)
            A = np.copy(self.A)

        return (Y, E, A)

    def plot_endmembers(
        self,
        E0=None,
        normalize=False,
        display=True,
        run=0,
    ):
        """
        Display endmembers spectrum signature
        """
        title = f"{self.shortname}"
        ylabel = "Reflectance"
        xlabel = "# Bands"
        if E0 is None:
            E = np.copy(self.scaledE)
            title += " GT Endmembers"
            linestyle = "-"
        else:
            assert self.E.shape == E0.shape
            E = np.copy(E0)
            title += " Estimated Endmembers"
            linestyle = "--"
        # if normalize:
        #     title += " ($l_\infty$-normalized)"
        #     ylabel += " Normalized"
        plt.figure(figsize=(12, 4))
        for pp in range(self.p):
            data = E[:, pp]
            # if normalize:
            #     data /= E[:, pp].max()
            plt.plot(data, label=self.labels[pp], linestyle=linestyle)
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if display:
            plt.show()
        else:
            figname = f"{self.shortname}-"
            # figname += "GT_" if E0 is None else ""
            figname += f"endmembers-{run}-"
            figname += "GT" if E0 is None else ""
            figname += ".png"
            plt.savefig(os.path.join(self.figs_dir, figname))
            plt.close()

    def plot_abundances(
        self,
        A0=None,
        grid=None,
        transpose=False,
        display=True,
        run=0,
    ):
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
                mappable = curr_ax.imshow(
                    A[kk, :, :],
                    vmin=0.0,
                    vmax=1.0,
                )
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                fig.colorbar(
                    mappable,
                    ax=curr_ax,
                    location="right",
                    shrink=0.5,
                )
                kk += 1

                if kk == self.p:
                    break

        plt.suptitle(title)
        if display:
            plt.show()
        else:
            figname = f"{self.shortname}-"
            figname += f"abundances-{run}-"
            figname += "GT" if A0 is None else ""
            figname += ".png"
            path = os.path.join(self.figs_dir, figname)
            plt.savefig(path)
            plt.close()

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
        display=True,
        run=0,
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
                mappable = curr_ax.imshow(
                    X[kk, :, :],
                    cmap="inferno",
                    # vmin=0.0,
                    # vmax=1.0,
                )
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                fig.colorbar(
                    mappable,
                    ax=curr_ax,
                    shrink=0.5,
                    location="right",
                )
                kk += 1

                if kk == self.p:
                    break

        plt.suptitle(title)
        if display:
            plt.show()
        else:
            filename = f"{self.shortname}_contributions-{run}.png"
            path = os.path.join(self.figs_dir, filename)
            plt.savefig(path)
            plt.close()

    def plot_PCA(
        self,
        E0=None,
        display=True,
        run=0,
        initializer=False,
    ):
        """
        Plot 2D PCA-projected data manifold
        """
        title = f"{self.shortname} 2D PCA-projected data manifold"
        xlabel = "PC #1"
        ylabel = "PC #2"
        # if E0 is None:
        #     E = np.copy(self.E)
        #     title += " - GT"
        # else:
        #     assert self.E.shape == E0.shape
        #     E = np.copy(E0)
        #     title += " - Estimated endmembers"

        logger.info("Computing SVD...")
        U, _, _ = LA.svd(self.Y, full_matrices=False)

        U1, U2 = U[0], U[1]

        y1, y2 = U1 @ self.Y, U2 @ self.Y
        e1, e2 = U1 @ self.scaledE, U2 @ self.scaledE

        plt.scatter(y1, y2, label="pixel")
        plt.scatter(e1, e2, label="GT endmember")
        if E0 is not None:
            assert self.scaledE.shape == E0.shape
            x1, x2 = U1 @ E0, U2 @ E0
            if initializer:
                plt.scatter(x1, x2, c="black", label="initializer")
            else:
                plt.scatter(x1, x2, c="red", label="Est. endmember")
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if display:
            plt.show()
        else:
            figname = f"{self.shortname}-"
            figname += f"PCA-{run}-"
            figname += "GT" if E0 is None or initializer else ""
            figname += ".png"
            plt.savefig(os.path.join(self.figs_dir, figname))
            plt.close()


if __name__ == "__main__":
    pass
