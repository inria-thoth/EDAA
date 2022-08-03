import logging
import os

import numpy as np
import scipy.io as sio
from hsi_unmixing.data.datasets.base import HSI
from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2
from hsi_unmixing.data.normalizers import RawInput
from hsi_unmixing.models.aligners import (CustomAbundancesAligner,
                                          MunkresAbundancesAligner,
                                          MunkresAligner)
from hsi_unmixing.models.metrics import (MeanAbsoluteError, MeanSquareError,
                                         RMSEAggregator, SADAggregator,
                                         SpectralAngleDistance)
from hsi_unmixing.utils import save_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# def show_abunds(dataset):
#     E, A = load_results(dataset)

#     h, w, p = A.shape
#     A = A.transpose((2, 0, 1))
#     A = A.reshape(p, h * w)

#     hsi = HSI("SamsonFixed.mat")


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
    # hsi = HSI("SamsonFixed.mat")
    # hsi = HSI("JasperRidge.mat", normalizer=RawInput())
    # hsi = HSI("Urban6.mat", normalizer=PL2())
    # hsi = HSI("TinyAPEX.mat", normalizer=PL2())
    # hsi = HSI("WDC.mat", normalizer=RawInput())
    hsi = HSI("WDC.mat", normalizer=PL2())
    # hsi = HSI("Samson.mat", normalizer=PL2())
    # hsi = HSI("JasperRidge.mat", normalizer=PL2())

    # hsi = HSI("JasperRidge.mat")

    # # Aligner
    # criterion = MeanAbsoluteError()
    # criterion = SpectralAngleDistance()
    # aligner = MunkresAligner(hsi=hsi, criterion=criterion)

    # Align data
    # E1 = aligner.fit_transform(E)
    # A1 = aligner.transform_abundances(A)
    criterion = MeanSquareError()
    aligner = MunkresAbundancesAligner(hsi=hsi, criterion=criterion)

    # Align data
    A1 = aligner.fit_transform(A)
    E1 = aligner.transform_endmembers(E)

    # P = np.zeros((p, p))
    # # P[0, 2] = 1
    # # P[1, 0] = 1
    # # P[2, 1] = 1
    # P[2, 0] = 1
    # P[0, 1] = 1
    # P[1, 2] = 1

    # # Custom alignment
    # aligner = CustomAbundancesAligner(hsi=hsi, criterion=criterion, P=P)

    # A1 = aligner.fit_transform(A)
    # E1 = aligner.transform_endmembers(E)

    # Add run
    RMSE.add_run(0, hsi.A, A1, hsi.labels)
    SAD.add_run(0, hsi.E, E1, hsi.labels)

    logger.debug("...Done")

    # Aggregate results
    RMSE.aggregate()
    SAD.aggregate()

    save_estimates(E1, A1, hsi)


if __name__ == "__main__":
    # main("Apex")
    main("WDC")
    # main("Urban6")
    # main("SamsonOld")
    # main("JasperRidge")
    # main("Urban4")
