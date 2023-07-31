import logging
import time

import numpy as np
import scipy.sparse as sp
import spams
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DecompSimplex:
    def __init__(self, T):
        self.T = T
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        p,
        *args,
        **kwargs,
    ):
        def loss(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def update_spamsB(a, b):
            R = Y - (Y @ b) @ a
            for ii in range(p):
                R = R + (Y @ b[:, ii])[:, np.newaxis] @ a[ii, :][np.newaxis, :]
                z = (R @ a[ii, :]) / (a[ii, :] @ a[ii, :].T)
                z = np.asfortranarray(z[:, np.newaxis])
                bb = spams.decompSimplex(z, YY)
                b[:, ii] = np.squeeze(bb.todense())
                R = R - (Y @ b[:, ii])[:, np.newaxis] @ a[ii, :][np.newaxis, :]
            return b

        tic = time.time()

        _, N = Y.shape

        YY = np.asfortranarray(Y)
        B = (1 / N) * np.ones((N, p))
        A = (1 / p) * np.ones((p, N))

        for _ in tqdm(range(self.T)):
            logger.debug(f"pre update B => {loss(A, B):.2f}")
            B = update_spamsB(A, B)
            logger.debug(f"post update B => {loss(A, B):.2f}")
            logger.debug(f"pre update A => {loss(A, B):.2f}")
            A = np.array(spams.decompSimplex(YY, np.asfortranarray(Y @ B)).todense())
            logger.debug(f"post update A => {loss(A, B):.2f}")

        tac = time.time()

        self.time = round(tac - tic, 2)

        logger.info(f"{self} took {self.time}s")

        return Y @ B, A


class ArchetypalAnalysis:
    def __init__(self, params: dict = {}):
        self.params = params

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        p,
        E0=None,
        *args,
        **kwargs,
    ):
        """
        Archetypal Analysis optimizer from SPAMS

        Parameters:
            Y: `numpy array`
                2D data matrix (L x N)

            p: `int`
                Number of endmembers

            E0: `numpy array`
                2D initial endmember matrix (L x p)
                Default: None

        Source: http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams004.html#sec8
        """
        tic = time.time()
        Yf = np.asfortranarray(Y, dtype=np.float64)
        Ef = E0
        if Ef is not None:
            assert p == E0.shape[1]
            Ef = np.asfortranarray(Ef, dtype=np.float64)

        Ehat, Asparse, Xsparse = spams.archetypalAnalysis(
            Yf,
            p=p,
            Z0=Ef,
            returnAB=True,
            **self.params,
        )

        self.E = Ehat
        self.A = sp.csc_matrix.toarray(Asparse)
        self.Xmap = sp.csc_matrix.toarray(Xsparse).T
        tac = time.time()
        self.time = round(tac - tic, 2)

        logger.info(f"{self} took {self.time}s")
        return self.E, self.A
