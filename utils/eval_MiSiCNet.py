import logging
import os

import scipy.io as sio
from hsi_unmixing.data.datasets.base import HSI
from hsi_unmixing.models.aligners import MunkresAligner
from hsi_unmixing.models.metrics import (MeanAbsoluteError, RMSEAggregator,
                                         SADAggregator)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def show_abunds(dataset):
    E, A = load_results(dataset)

    h, w, p = A.shape
    A = A.transpose((2, 0, 1))
    A = A.reshape(p, h * w)

    hsi = HSI("SamsonFixed.mat")


def load_results(dataset):
    dir_path = f"/home/azouaoui/github/MiSiCNet/Results/{dataset}"
    E_path = os.path.join(dir_path, "E.mat")
    A_path = os.path.join(dir_path, "A.mat")
    E = sio.loadmat(E_path)["E"]
    A = sio.loadmat(A_path)["A"]
    return E, A


def main(dataset):
    RMSE = RMSEAggregator()
    SAD = SADAggregator()

    E, A = load_results(dataset)

    # breakpoint()

    # for run, path in enumerate(os.listdir(dir_path)):
    #     full_path = os.path.join(dir_path, path)
    #     logger.debug(f"{full_path} to be opened...")
    #     data = sio.loadmat(full_path)
    #     E = data["M"]
    #     A = data["A"]

    #     # Reshape data
    #     E = E.T
    h, w, p = A.shape
    A = A.transpose((2, 0, 1))
    A = A.reshape(p, h * w)

    # Get HSI
    # hsi = HSI("TinyAPEX.mat")
    # hsi = HSI("WDC.mat")
    # hsi = HSI("Urban4.mat")
    hsi = HSI("SamsonFixed.mat")

    # Aligner
    criterion = MeanAbsoluteError()
    aligner = MunkresAligner(hsi=hsi, criterion=criterion)

    # Align data
    E1 = aligner.fit_transform(E)
    A1 = aligner.transform_abundances(A)

    # Add run
    RMSE.add_run(0, hsi.A, A1, hsi.labels)
    SAD.add_run(0, hsi.E, E1, hsi.labels)

    logger.debug("...Done")

    # Aggregate results
    RMSE.aggregate()
    SAD.aggregate()


if __name__ == "__main__":
    # main("Apex")
    # main("WDC")
    # main("Urban4")
    main("Samson")
