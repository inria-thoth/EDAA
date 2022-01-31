import logging
import pdb
import time

import numpy as np
import scipy.sparse as sp
import spams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DecompSimplex:
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        E,
        computeXtX=True,
        numThreads=-1,
    ):
        """
        Active-set algorithm from SPAMS to solve Sparse Coding under simplex constraints


        Parameters:
            Y: `numpy array`
                2D data matrix (L x N).

            E: `numpy array`
                2D matrix of endmembers (L x p).

        Returns:
            A: `numpy array`
                2D abundance maps (p x N).




        Reference: http://thoth.inrialpes.fr/people/mairal/spams/doc-python/html/doc_spams005.html#sec22
        """
        tic = time.time()
        Yf = np.asfortranarray(Y, dtype=np.float64)
        Ef = np.asfortranarray(E, dtype=np.float64)

        W = spams.decompSimplex(
            Yf,
            Ef,
            computeXtX=computeXtX,
            numThreads=numThreads,
        )

        A = sp.csr_matrix.toarray(W)
        tac = time.time()
        logger.info(f"{self} took {tac - tic:.2f}s")

        return A
