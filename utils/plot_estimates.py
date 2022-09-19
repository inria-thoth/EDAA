import logging
import os

import matplotlib.pyplot as plt
import scipy.io as sio
from hsi_unmixing.utils import load_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(path: str):
    # hsi = HSI("Samson.mat")
    print(f"Loading => {path}")
    data = load_estimates(path)
    # breakpoint()
    plot_endmembers(data["Ehat"], data["Egt"])
    plot_abundances(
        data["Ahat"],
        data["Agt"],
        int(data["H"]),
        int(data["W"]),
        transpose=False,
    )


def plot_endmembers(Ehat, Egt):
    assert Ehat.shape == Egt.shape
    p = Ehat.shape[1]

    fig, axes = plt.subplots(nrows=p, ncols=1, figsize=(6 * p, 6))

    for ii in range(p):
        ax = axes[ii]
        ax.plot(Egt[:, ii])
        ax.plot(Ehat[:, ii])

    plt.show()


def plot_abundances(Ahat, Agt, H, W, transpose=False):
    assert Ahat.shape == Agt.shape

    p = Ahat.shape[0]

    Ahat = Ahat.reshape(p, H, W)
    Agt = Agt.reshape(p, H, W)

    if transpose:
        Ahat = Ahat.transpose(0, 2, 1)
        Agt = Agt.transpose(0, 2, 1)

    fig, axes = plt.subplots(nrows=p, ncols=2, figsize=(6 * p, 12))

    for ii in range(p):
        ax1 = axes[ii, 0]
        ax2 = axes[ii, 1]

        ax1.imshow(Agt[ii, :, :], vmin=0.0, vmax=1.0)
        mappable = ax2.imshow(Ahat[ii, :, :], vmin=0.0, vmax=1.0)
        ax1.axis("off")
        ax2.axis("off")

        fig.colorbar(mappable, ax=ax2, location="right", shrink=0.5)

    plt.show()


if __name__ == "__main__":
    run_path = os.path.join("data", "runs", "2022-08-04_16-39-25", "5")
    # run_path = os.getcwd()
    file_path = os.path.join(run_path, "estimates.mat")
    main(file_path)
