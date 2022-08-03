import logging
import os

import scipy.io as sio
from hsi_unmixing.data.datasets.base import HSI
from hsi_unmixing.data.normalizers import PixelwiseL2Norm as PL2
from hsi_unmixing.data.normalizers import RawInput
from hsi_unmixing.models.aligners import (MunkresAbundancesAligner,
                                          MunkresAligner)
from hsi_unmixing.models.metrics import (MeanAbsoluteError, MeanSquareError,
                                         RMSEAggregator, SADAggregator)
from hsi_unmixing.utils import save_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(dataset):
    RMSE = RMSEAggregator()
    SAD = SADAggregator()

    dir_path = f"/home/azouaoui/github/hu_autoencoders/Results/Endnet/{dataset}"
    for run, path in enumerate(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, path)
        if ".mat" not in full_path:
            continue
        logger.debug(f"{full_path} to be opened...")
        data = sio.loadmat(full_path)
        E = data["M"]
        A = data["A"]

        # Reshape data
        E = E.T
        h, w, p = A.shape
        A = A.transpose((2, 0, 1))
        A = A.reshape(p, h * w)

        # Get HSI
        # hsi = HSI(f"{dataset}.mat")
        # hsi = HSI("Urban4.mat")
        # hsi = HSI("Samson.mat", normalizer=PL2())
        # hsi = HSI("JasperRidge.mat", normalizer=PL2())
        # hsi = HSI("Samson.mat", normalizer=RawInput())
        # hsi = HSI("Urban6.mat", normalizer=PL2())
        hsi = HSI("WDC.mat", normalizer=PL2())
        # hsi = HSI("TinyAPEX.mat", normalizer=PL2())
        # hsi = HSI("TinyAPEX.mat", normalizer=RawInput())
        # hsi = HSI("WDC.mat", normalizer=RawInput())
        # hsi = HSI("TinyAPEX.mat")
        # hsi = HSI("WDC.mat")

        # Aligner
        criterion = MeanAbsoluteError()
        aligner = MunkresAligner(hsi=hsi, criterion=criterion)

        # Align data
        E1 = aligner.fit_transform(E)
        A1 = aligner.transform_abundances(A)

        # criterion = MeanSquareError()
        # aligner = MunkresAbundancesAligner(hsi=hsi, criterion=criterion)

        # A1 = aligner.fit_transform(A)
        # E1 = aligner.transform_endmembers(E)

        # Add run
        RMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)

        logger.debug("...Done")

    # Aggregate results
    RMSE.aggregate()
    SAD.aggregate()

    save_estimates(E1, A1, hsi)


if __name__ == "__main__":
    # main("TinyAPEX")
    main("WDC")
    # main("Urban")
    # main("U6")
    # main("Urban")
    # main("Samson")
    # main("JasperRidge")
    # main("Samson")
