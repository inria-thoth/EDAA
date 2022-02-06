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


if __name__ == "__main__":
    from hsi_unmixing.data.datasets.base import HSI
    from hsi_unmixing.models.aligners import GreedyAligner as GA
    from hsi_unmixing.models.initializers import VCA

    hsi = HSI("Samson.mat")

    vca = VCA()
    Einit = vca.init_like(hsi)
    solver = ArchetypalAnalysis()
    E0, A0 = solver.solve(hsi.Y, hsi.p, E0=Einit)

    aligner = GA(hsi, "MeanAbsoluteError")
    Ehat = aligner.fit_transform(E0)
    Ahat = aligner.transform_abundances(A0)

    # hsi.plot_abundances(transpose=True)
    # hsi.plot_abundances(transpose=True, A0=Ahat)

    # hsi.plot_endmembers()
    # hsi.plot_endmembers(E0=Ehat)

    Xhat = aligner.transform_abundances(solver.Xmap)
    hsi.plot_contributions(transpose=True, X0=Xhat, method=solver)
