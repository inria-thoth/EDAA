import logging

from hydra.utils import instantiate
from hydra.utils import to_absolute_path

# import matlab.engine

import numpy as np
import numpy.linalg as LA
import scipy.io as sio
from munkres import Munkres

from hsi_unmixing.data.noises import AdditiveWhiteGaussianNoise as AWGN
from hsi_unmixing.models.metrics import ARMSEAggregator, SADAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):

    ARMSE = ARMSEAggregator()
    SAD = SADAggregator()

    labels = [f"#{ii}" for ii in range(6)]
    # Load data
    data_path = to_absolute_path(f"./data/MixedRatio_rho{cfg.rho}.mat")
    data = sio.loadmat(data_path)
    logger.debug(data.keys())

    Y = data["Y"]
    E_gt = data["E"]
    A_gt = data["A"]
    p = 6

    logger.debug(f"Y shape => {Y.shape}")
    logger.debug(f"Y min => {Y.min()}, Y max => {Y.max()}")
    logger.debug(f"E shape => {E_gt.shape}")
    logger.debug(f"A shape => {A_gt.shape}")

    noise = AWGN()
    SNR = cfg.SNR

    # Instantiate aligner
    MSE = MeanSquareError()
    aligner = MunkresAbundancesAligner(Aref=A_gt, criterion=MSE)

    for run in range(cfg.runs):
        # Apply noise at each run
        seed = cfg.seed + run
        Y_noisy = noise.fit_transform(Y=Y, SNR=SNR, seed=seed)
        Y_noisy = np.clip(Y_noisy, 0.0, 1.0)
        logger.debug(f"Y noisy shape => {Y_noisy.shape}")

        # Apply L2 pixelwise normalization
        Y_noisy_normalized = Y_noisy / np.linalg.norm(Y_noisy, axis=0, keepdims=True)
        E_gt_normalized = E_gt / np.linalg.norm(E_gt, axis=0, keepdims=True)

        # Load new model at each run
        model = instantiate(cfg.model)

        E0, A0 = model.solve(Y_noisy_normalized, 6, seed=seed, H=40, W=25)

        # Align output
        A1 = aligner.fit_transform(A0)
        E1 = aligner.transform_endmembers(E0)

        # Compute metrics

        ARMSE.add_run(run, A_gt, A1, labels)
        SAD.add_run(run, E_gt_normalized, E1, labels)

    ARMSE.aggregate()
    SAD.aggregate()

    # NOTE Save last estimates


class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class MeanSquareError(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.sqrt(normE.T**2 + normEref**2 - 2 * (E.T @ Eref))


class BaseAbundancesAligner:
    def __init__(self, Aref, criterion):
        self.Aref = Aref
        self.criterion = criterion
        self.P = None
        self.dists = None

    def fit(self, A):
        raise NotImplementedError

    def transform(self, A):
        assert self.P is not None, "Must be fitted first"
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]

        return self.P @ A

    def transform_endmembers(self, E):
        assert self.P is not None, "Must be fitted first"
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]

        return E @ self.P.T

    def fit_transform(self, A):

        self.fit(A)
        res = self.transform(A)
        return res

    def __repr__(self):
        msg = f"{self.__class__.__name__}_crit{self.criterion}"
        return msg


class MunkresAbundancesAligner(BaseAbundancesAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, A):

        # Computing distance matrix
        self.dists = self.criterion(A.T, self.Aref.T)

        # Initialization
        p = A.shape[0]
        P = np.zeros((p, p))

        m = Munkres()
        indices = m.compute(self.dists)
        for row, col in indices:
            P[row, col] = 1.0

        self.P = P.T
