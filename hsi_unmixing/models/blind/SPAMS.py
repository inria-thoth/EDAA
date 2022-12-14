import logging
import pdb
import time

import numpy as np
import scipy.sparse as sp
import spams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
