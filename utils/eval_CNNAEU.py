import logging
import os

import scipy.io as sio
from hsi_unmixing.data.datasets.base import HSI
from hsi_unmixing.models.aligners import MunkresAligner
from hsi_unmixing.models.metrics import (MeanAbsoluteError, RMSEAggregator,
                                         SADAggregator)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(dataset):
    RMSE = RMSEAggregator()
    SAD = SADAggregator()

    dir_path = f"/home/azouaoui/github/hu_autoencoders/Results/CNNAEU/{dataset}"
    for run, path in enumerate(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, path)
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
        hsi = HSI(f"{dataset}.mat")

        # Aligner
        criterion = MeanAbsoluteError()
        aligner = MunkresAligner(hsi=hsi, criterion=criterion)

        # Align data
        E1 = aligner.fit_transform(E)
        A1 = aligner.transform_abundances(A)

        # Add run
        RMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)

        logger.debug("...Done")

    # Aggregate results
    RMSE.aggregate()
    SAD.aggregate()


if __name__ == "__main__":
    main("Samson")
