import logging
import pdb
import time

# import hsi_unmixing.models.metrics as criterions
import numpy as np
from hungarian_algorithm import algorithm as HA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseAligner:
    def __init__(self, hsi, criterion):
        self.Eref = hsi.E
        # self.criterion = criterions.__dict__[criterion]()
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
        tic = time.time()

        self.fit(E)
        res = self.transform(E)

        tac = time.time()
        logger.info(f"{self} took {tac - tic:.2f}s...")

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


class HungarianAlgorithmAligner(BaseAligner):
    def __init__(self, hsi, **kwargs):
        super().__init__(hsi=hsi, **kwargs)
        self.labels = {
            str(ii): label
            for ii, label in enumerate(
                hsi.labels,
            )
        }
        self.reverse_labels = {v: k for k, v in self.labels.items()}

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

        Source: https://github.com/benchaplin/hungarian-algorithm
        """

        self.dists = self.criterion(E, self.Eref)

        # Initialization
        self.P = None
        # Create graph for hungarian algorithm (HA)
        G = self.create_graph()
        # pdb.set_trace()
        # Find matching
        results = HA.find_matching(
            G,
            matching_type="min",
            return_type="list",
        )
        # Convert matching to permutation matrix
        self.matching2matrix(results)

    def create_graph(self):
        p = self.dists.shape[0]
        assert p == self.dists.shape[1]
        G = {
            str(ii): {self.labels[str(jj)]: self.dists[ii, jj] for jj in range(p)}
            for ii in range(p)
        }
        return G

    def matching2matrix(self, results):
        """
        Build a permutation matrix based on the HA results list output
        """
        # Initialization
        p = len(results)
        P = np.zeros((p, p))

        for result in results:
            pair, weight = result
            _from, _to_label = pair
            P[int(_from), int(self.reverse_labels[_to_label])] = 1.0

        self.P = P


if __name__ == "__main__":

    from hsi_unmixing.data.datasets.base import HSI
    from hsi_unmixing.models.metrics import MeanAbsoluteError as MAE

    hsi = HSI("Samson.mat", figs_dir=None)
    Eref = hsi.E
    L, p = Eref.shape

    generator = np.random.RandomState(seed=0)
    Q = generator.permutation(np.eye(p))
    E = Eref @ Q + 0.001 * generator.randn(L, p)

    # metric = "MeanAbsoluteError"

    criterion = MAE()

    for cls in [
        NoneAligner,
        GreedyAligner,
        HungarianAlgorithmAligner,
    ]:

        aligner = cls(hsi=hsi, criterion=criterion)
        Ehat = aligner.fit_transform(E)

        print(f"{aligner}")
        print("-" * 15)
        print(f"Generated permutation:\n{Q}")
        print(f"Estimated permutation:\n{aligner.P}")
        print(f"Transposed estimated permutation:\n{aligner.P.T}")
