import logging
import pdb

import numpy as np
import numpy.linalg as LA
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseMetric:
    def __init__(self):
        pass

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        # Expect => L (# HSI channels) x p (# endmembers)
        # assert E.shape[0] > E.shape[1]
        assert type(X) == type(Xref)

        if isinstance(X, torch.Tensor):
            logger.debug("Convert tensors to arrays in Metric class...")
            X = X.detach().numpy()
            Xref = Xref.detach().numpy()

        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"


class MeanAbsoluteError(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return 100 * (1 - np.abs((E / normE).T @ (Eref / normEref)))


class SpectralAngleDistance(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.arccos((E / normE).T @ (Eref / normEref))


class MeanSquareError(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.sqrt(normE.T ** 2 + normEref ** 2 - 2 * (E.T @ Eref))


class aRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        A, Aref = self._check_input(A, Aref)

        # Expect abundances: p (# endmembers) x N (# pixels)
        assert A.shape[0] < A.shape[1]

        # return 100 * np.sqrt(((A - Aref) ** 2).mean(0)).mean(0)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())
