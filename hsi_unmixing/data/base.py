import logging
import os
import time
import pdb

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import spams


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HSI:
    def __init__(
        self,
        name: str,
        data_path: str = "./data",
    ):
        path = os.path.join(data_path, name)
        logger.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path)
        self.shortname = name.strip(".mat")

        data = sio.loadmat(path)
        logger.debug(f"Data keys: {data.keys()}")

        for key in filter(
            lambda s: not s.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(key, data[key])

        self.dtype = np.float64
        # Data format check
        self.H = self.H.item()
        self.W = self.W.item()
        self.L = self.L.item()
        self.p = self.p.item()

        self.N = self.H * self.W

        # pdb.set_trace()
        assert self.E.shape == (self.L, self.p)
        assert self.Y.shape == (self.L, self.N)

        # Normalize Y
        self.Y = (self.Y - self.Y.min()) / (self.Y.max() - self.Y.min())

        try:
            assert len(self.labels) == self.p
        except AssertionError:
            # Create pseudo labels
            self.labels = np.arange(self.p)
        # Create GT abundances if not existing

        try:
            self.__getattribute__("A")
        except AttributeError:
            self.get_GT_abundances()

        assert self.A.shape == (self.p, self.N)
        assert np.allclose(self.A.sum(0), np.ones(self.N))

    def __repr__(self):
        msg = f"HSI => {self.shortname}\n"
        msg += "---------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples, ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({list(self.labels)})\n"
        msg += f"MinValue: {self.Y.min()}, MaxValue: {self.Y.max()}"
        return msg

    def get_GT_abundances(self):
        """
        Compute GT abundances based on GT endmembers using decompSimplex
        """
        Yf = np.asfortranarray(self.Y, dtype=self.dtype)
        Ef = np.asfortranarray(self.E, dtype=self.dtype)

        tic = time.time()
        W = spams.decompSimplex(Yf, Ef, computeXtX=True)
        tac = time.time()

        logging.info(f"Computing GT abundances took {tac - tic:.2f}s")

        self.A = sp.csr_matrix.toarray(W)
