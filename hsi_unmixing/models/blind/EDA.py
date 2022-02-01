import logging
import pdb
import time

# import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EDA:
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        p,
        E,
        eta0=1.0,
        K=100,
        tol=1e-6,
        alpha_init="softmax",
        steps="sqrt-simple",
    ):
        """
        Entropic Descent Algorithm

        Parameters:
            Y: `torch Tensor`
                2D data matrix (L x N)

            p: `int`
                Number of endmembers

            E0: `torch Tensor`
                2D initial endmember matrix (L x p)

            eta0: `float`
                Step size initial value

            K: `int`
                Maximum number of unfoldings

            tol: `float`
                Stopping criterion

            alpha_init: `str`
                Alpha initialization enforcing ASC

            steps: `str`
                Step sizes decreasing trend
        """
        tic = time.time()
        # Sanity checks
        L, N = Y.shape
        assert L == E.shape[0]
        assert p == E.shape[1]

        Y = Y.t()
        # E = E.t()

        # Step sizes scheme
        if steps == "sqrt-simple":
            etas = eta0 / torch.sqrt(torch.arange(start=1, end=K + 1))
        elif steps == "flat":
            etas = eta0 * torch.ones(K)
        else:
            raise NotImplementedError

        # Inner functions
        def f(alpha):
            return 0.5 * (Y - F.linear(alpha, E) ** 2).mean()

        def grad_f(alpha):
            return -F.linear(Y - F.linear(alpha, E), E.t())

        # Initialization
        if alpha_init == "softmax":
            alpha = F.softmax(F.linear(Y, E.t()), dim=1)
        elif alpha_init == "uniform":
            alpha = torch.ones(N, p) / p
        else:
            raise NotImplementedError

        logger.info(f"Loss: {f(alpha):.6f} [0]")

        # Encoding
        for kk in range(K):
            alpha = self.update(alpha, -etas[kk] * grad_f(alpha))
            logger.info(f"Loss: {f(alpha):.6f} [{kk+1}]")

        # Decoding
        Y_hat = F.linear(alpha, E)

        tac = time.time()

        logger.info(f"{self} took {tac - tic:.2f}s")

        return Y_hat.t(), E, alpha.t()

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
        "K": 300,
        "alpha_init": "softmax",
        "steps": "sqrt-simple",
        "eta0": 0.1,
    }

    vca = VCA()
    te = TrueEndmembers()
    # Einit = vca.init_like(hsi)
    Einit = te.init_like(hsi)
    solver = EDA()
    Yt, Et, At = hsi(asTensor=True)
    Y0, E0, A0 = solver.solve(
        Yt,
        hsi.p,
        E=torch.Tensor(Einit),
        **params,
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
