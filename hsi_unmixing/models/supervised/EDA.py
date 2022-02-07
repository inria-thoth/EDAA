import logging
import pdb
import time

# import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EDA:
    def __init__(
        self,
        eta0=0.1,
        K=1000,
        alpha_init="softmax",
        steps="sqrt-simple",
    ):
        """
        eta0: `float`
            Step size initial value

        K: `int`
            Number of unfoldings

        alpha_init: `str`
            Alpha initialization enforcing ASC

        steps: `str`
            Step sizes decreasing trend
        """
        self.eta0 = eta0
        self.K = K
        self.alpha_init = alpha_init
        self.steps = steps

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        E0,
    ):
        """
        Entropic Descent Algorithm

        Parameters:
            Y: `torch Tensor`
                2D data matrix (L x N)

            E0: `torch Tensor`
                2D initial endmember matrix (L x p)

        """
        tic = time.time()
        # Sanity checks
        L, N = Y.shape
        assert L == E0.shape[0]
        p = E0.shape[1]

        Y = Y.t()

        # Step sizes scheme
        if self.steps == "sqrt-simple":
            etas = self.eta0 / torch.sqrt(torch.arange(start=1, end=self.K + 1))
        elif self.steps == "flat":
            etas = self.eta0 * torch.ones(self.K)
        else:
            raise NotImplementedError

        # Inner functions
        def f(alpha):
            return 0.5 * ((Y - F.linear(alpha, E0)) ** 2).sum()

        def grad_f(alpha):
            return -F.linear(Y - F.linear(alpha, E0), E0.t())

        # Initialization
        if self.alpha_init == "softmax":
            alpha = F.softmax(F.linear(Y, E0.t()), dim=1)
        elif self.alpha_init == "uniform":
            alpha = torch.ones(N, p) / p
        else:
            raise NotImplementedError

        logger.debug(f"Loss: {f(alpha):.6f} [0]")

        # Encoding
        for kk in range(self.K):
            alpha = self.update(alpha, -etas[kk] * grad_f(alpha))
            logger.debug(f"Loss: {f(alpha):.6f} [{kk+1}]")

        # Decoding
        Y_hat = F.linear(alpha, E0)

        tac = time.time()

        logger.info(f"{self} took {tac - tic:.2f}s")

        self.A = alpha.t()

        return self.A

    @staticmethod
    def update(a, b):
        m, _ = torch.max(b, dim=1, keepdim=True)
        num = a * torch.exp(b - m)
        denom = torch.sum(num, dim=1, keepdim=True)
        return num / denom


if __name__ == "__main__":

    from hsi_unmixing.data.datasets.base import HSI
    from hsi_unmixing.models.aligners import GreedyAligner as GA
    from hsi_unmixing.models.initializers import VCA, TrueEndmembers

    hsi = HSI("JasperRidge.mat")

    params = {
        "K": 1000,
        "alpha_init": "softmax",
        "steps": "sqrt-simple",
        "eta0": 0.1,
    }

    vca = VCA()
    te = TrueEndmembers()
    # Einit = vca.init_like(hsi)
    Einit = te.init_like(hsi)
    solver = EDA(**params)
    Yt, Et, At = hsi(asTensor=True)
    Y0, E0, A0 = solver.solve(
        Yt,
        E0=torch.Tensor(Einit),
    )

    E0 = E0.detach().numpy()
    A0 = A0.detach().numpy()

    aligner = GA(hsi, "MeanAbsoluteError")
    Ehat = aligner.fit_transform(E0)
    Ahat = aligner.transform_abundances(A0)

    hsi.plot_abundances(transpose=True)
    hsi.plot_abundances(transpose=True, A0=Ahat)

    hsi.plot_endmembers()
    hsi.plot_endmembers(E0=Ehat)
