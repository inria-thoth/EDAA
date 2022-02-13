import logging
import pdb

import numpy as np
import numpy.linalg as LA
import pandas as pd

# import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        # Expect => L (# HSI channels) x p (# endmembers)
        # assert E.shape[0] > E.shape[1]
        assert type(X) == type(Xref)

        # if isinstance(X, torch.Tensor):
        #     logger.debug("Convert tensors to arrays in Metric class...")
        #     X = X.detach().numpy()
        #     Xref = Xref.detach().numpy()

        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


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


class SADDegrees(SpectralAngleDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        tmp = super().__call__(E, Eref)
        return (np.diag(tmp) * (180 / np.pi)).mean()


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
        # assert A.shape[0] < A.shape[1]

        # return 100 * np.sqrt(((A - Aref) ** 2).mean(0)).mean(0)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())


class RunAggregator:
    def __init__(
        self,
        metric,
        use_endmembers=False,
    ):
        """
        Aggregate runs by tracking a metric
        """
        self.metric = metric
        self.use_endmembers = use_endmembers
        self.filename = f"{metric}.json"
        # self.A_metrics = {"RMSE": aRMSE()}
        # self.E_metrics = {"SAD": SADDegrees()}
        self.data = {}
        self.df = None
        self.summary = None

    def add_run(self, run, X, Xhat, labels):

        d = {}
        d["Overall"] = self.metric(X, Xhat)
        for ii, label in enumerate(labels):
            if self.use_endmembers:
                x, xhat = X[:, ii][:, None], Xhat[:, ii][:, None]
                d[label] = self.metric(x, xhat)
            else:
                d[label] = self.metric(X[ii], Xhat[ii])

        logger.debug(f"Run {run}: {self.metric} => {d}")

        self.data[run] = d

    def aggregate(self):
        self.df = pd.DataFrame(self.data).T
        self.summary = self.df.describe().round(2)
        logger.info(f"{self.metric} summary:\n{self.summary}")
        self.save()

    def save(self):
        self.df.to_json(f"runs-{self.filename}")
        self.summary.to_json(f"summary-{self.filename}")


class SADAggregator(RunAggregator):
    def __init__(self):
        super().__init__(SADDegrees(), use_endmembers=True)


class RMSEAggregator(RunAggregator):
    def __init__(self):
        super().__init__(aRMSE(), use_endmembers=False)
