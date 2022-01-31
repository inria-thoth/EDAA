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
    def _check_input(E, Eref):
        assert E.shape == Eref.shape
        # Expect => L (# HSI channels) x p (# endmembers)
        assert E.shape[0] > E.shape[1]
        assert type(E) == type(Eref)

        if isinstance(E, torch.Tensor):
            logger.debug("Convert tensors to arrays in Metric class...")
            E = E.detach().numpy()
            Eref = Eref.detach().numpy()

        return E, Eref

    def __call__(self, E, Eref):
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
