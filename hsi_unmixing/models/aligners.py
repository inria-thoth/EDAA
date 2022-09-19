import logging
import pdb

import numpy as np
from munkres import Munkres

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseAligner:
    def __init__(self, hsi, criterion):
        self.Eref = hsi.E
        self.criterion = criterion
        self.P = None
        self.dists = None

    def fit(self, E):
        raise NotImplementedError

    def transform(self, E):
        assert self.P is not None, "Must be fitted first"
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]

        return E @ self.P

    def transform_abundances(self, A):
        assert self.P is not None, "Must be fitted first"
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]

        return self.P.T @ A

    def fit_transform(self, E):

        self.fit(E)
        res = self.transform(E)

        return res

    def __repr__(self):
        msg = f"{self.__class__.__name__}_crit{self.criterion}"
        return msg


class NoneAligner(BaseAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, E):
        """
        Do not perform any alignment
        """
        self.P = np.eye(E.shape[1])


class GreedyAligner(BaseAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, E):
        """
        Greedy alignment based on distances matrix using specified criterion

        Parameters:
            E: `numpy array`
                2D matrix of endmembers (L x p)

        Records:
            dists: `numpy array`
                2D distance matrix (p x p)

            P: `numpy array`
                2D permutation matrix (p x p)
            Permutes the columns to align the endmembers
            according to ground truth
        """
        self.dists = self.criterion(E, self.Eref)

        # Initialization
        d = np.copy(self.dists)
        p = E.shape[1]
        P = np.zeros((p, p))

        for _ in range(p):
            # Select argmin value
            idx = np.unravel_index(d.argmin(), d.shape)
            # Assign selection to permutation matrix
            P[idx] = 1.0
            # Render corresponding row/col unpickable
            d[idx[0]] = np.inf
            d[:, idx[1]] = np.inf

        self.P = P


class MunkresAligner(BaseAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, E):
        """
        Alignment based on distances matrix using Hungarian Algorithm

        Parameters:
            E: `numpy array`
                2D matrix of endmembers (L x p)

        Records:
            dists: `numpy array`
                2D distance matrix between estimated and GT endmembers (p x p)

            P: `numpy array`
                2D permutation matrix (p x p)
            Permutes the columns to align the endmembers
            according to ground truth

        Source: https://software.clapper.org/munkres/
        """

        # Computing distance matrix
        self.dists = self.criterion(E, self.Eref)

        # Initialization
        p = E.shape[1]
        P = np.zeros((p, p))
        self.P = None

        m = Munkres()
        indexes = m.compute(self.dists)
        for row, col in indexes:
            P[row, col] = 1.0

        self.P = P


class BaseAbundancesAligner:
    def __init__(self, hsi, criterion):
        self.Aref = hsi.A
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


if __name__ == "__main__":
    pass
