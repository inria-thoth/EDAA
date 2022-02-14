import cProfile
import logging
import pdb
import time

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
            Maximum number of unfoldings

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
        p,
        E0=None,
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

        """
        tic = time.time()
        # Sanity checks
        L, N = Y.shape
        assert L == E0.shape[0]
        assert p == E0.shape[1]

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
            return 0.5 * (Y - F.linear(alpha, E0) ** 2).mean()

        def grad_f(alpha):
            return -F.linear(Y - F.linear(alpha, E0), E0.t())

        # Initialization
        if self.alpha_init == "softmax":
            alpha = F.softmax(F.linear(Y, E0.t()), dim=1)
        elif self.alpha_init == "uniform":
            alpha = torch.ones(N, p) / p
        else:
            raise NotImplementedError

        logger.info(f"Loss: {f(alpha):.6f} [0]")

        # Encoding
        for kk in range(self.K):
            alpha = self.update(alpha, -etas[kk] * grad_f(alpha))
            logger.info(f"Loss: {f(alpha):.6f} [{kk+1}]")

        # Decoding
        Y_hat = F.linear(alpha, E0)

        tac = time.time()

        logger.info(f"{self} took {tac - tic:.2f}s")

        return Y_hat.t(), E0, alpha.t()

    @staticmethod
    def update(a, b):
        m, _ = torch.max(b, dim=1, keepdim=True)
        num = a * torch.exp(b - m)
        denom = torch.sum(num, dim=1, keepdim=True)
        return num / denom


class AlternatingEDA:
    def __init__(
        self,
        eta0A=0.1,
        eta0B=0.1,
        KA=500,
        KB=50,
        A_init="softmax",
        schemeA="sqrt-simple",
        schemeB="sqrt-simple",
        nb_alternating=10,
        device=None,
        use_projection=False,
    ):
        """
        eta0A: `float`
            Step size initial value for Alpha

        eta0B: `float`
            Step size initial value for Beta

        K: `int`
            Maximum number of unfoldings

        A_init: `str`
            A initialization enforcing ASC

        schemeA: `str`
            Step sizes decreasing trend for A

        schemeB: `str`
            Step sizes decreasing trend for B
        """
        self.KA = KA
        self.KB = KB
        self.A_init = A_init
        self.etasA = self.get_steps_from_scheme(eta0A, schemeA, KA)
        self.etasB = self.get_steps_from_scheme(eta0B, schemeB, KB)
        self.nb_alternating = nb_alternating
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.use_projection = use_projection

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
        """
        Alternating Entropic Descent Algorithm

        Solves:
                    min        1 / 2 ||Y - YBA||^2_F
        alpha_i in delta_p for 1 <= i <= n
        beta_j in delta_n for 1 <= j <= p

        Parameters:
            Y: `torch Tensor`
                2D data matrix (L x N)

            p: `int`
                Number of endmembers / dimension of beta/alpha
        """
        tic = time.time()
        # Sanity checks
        L, N = Y.shape

        # YtY = Y.t() @ Y
        # Id = torch.eye(N)

        # Compute projection to (p - 1)-subspace here
        if self.use_projection:
            pdb.set_trace()
            logger.debug(f"Y shape before projection: {Y.shape}")
            # center Y
            meanY = Y.mean(1, keepdims=True)
            Y -= meanY
            diagY = Y @ Y.t() / N
            U = torch.linalg.svd(diagY, full_matrices=False)[0][:, : p - 1]
            Y = U.t() @ Y
            logger.debug(f"Y shape after projection: {Y.shape}")

        # Inner functions
        def f(a, b):
            return 0.5 * ((Y - Y @ b @ a) ** 2).sum()

        def grad_A(a, b):
            return -b.t() @ Y.t() @ (Y - Y @ b @ a)
            # return -b.t() @ YtY @ (Id - b @ a)

        def grad_B(a, b):
            # return -Y.t() @ Y @ a.t() + Y.t() @ Y @ b @ a @ a.t()
            return -Y.t() @ (Y - Y @ b @ a) @ a.t()
            # return -YtY @ (Id - b @ a) @ a.t()

        # Initialization
        # B = F.softmax(torch.ones(N, p) + 0.01 * torch.randn(N, p), dim=0)
        B = F.softmax(torch.randn(N, p), dim=0)
        # B = torch.eye(N, m=p)
        # B = torch.ones(N, p) / N
        # indices = torch.randint(0, high=N - 1, size=(p,))
        # E = Y[:, indices]

        # B = torch.linalg.pinv(Y) @ E
        # B = F.softmax(torch.linalg.pinv(Y) @ E, dim=0)

        if self.A_init == "softmax":
            A = F.softmax(B.t() @ Y.t() @ Y, dim=0)
        elif self.A_init == "uniform":
            A = torch.ones(p, N) / p
        else:
            raise NotImplementedError

        logger.debug(f"Initial loss: {f(A, B):.6f}")
        # print(f"Loss: {f(A, B):.6f} [0]")

        # To device
        Y = Y.to(self.device)
        A = A.to(self.device)
        B = B.to(self.device)

        etasA = self.etasA.to(self.device)
        etasB = self.etasB.to(self.device)
        # eta = etasB[0]

        with torch.no_grad():
            # Encoding
            for ii in range(self.nb_alternating):
                if ii % 2 == 0:
                    for kk in range(self.KA):
                        A = self.update(
                            A,
                            # -self.etasA[kk] / (ii + 1) * grad_A(A, B),
                            # -self.etasA[kk] * grad_A(A, B),
                            -etasA[kk] * grad_A(A, B),
                        )
                        # if kk == 0:
                        #     pass
                        logger.debug(f"Loss: {f(A, B):.6f} [{ii}|{kk+1}]")
                        # print(f"Loss: {f(A, B):.6f} [{ii}|{kk+1}]")

                else:
                    for kk in range(self.KB):
                        B = self.update(
                            B,
                            # -self.etasB[kk] / ii * grad_B(A, B),
                            # -self.etasB[kk] * grad_B(A, B),
                            -etasB[kk] * grad_B(A, B),
                            # -eta * grad_B(A, B),
                        )

                        # if kk == 0:
                        #     pass
                        logger.debug(f"Loss: {f(A, B):.6f} [{ii}|{kk+1}]")
                        # print(f"Loss: {f(A, B):.6f} [{ii}|{kk+1}]")

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"{self} took {self.time:.2f}s")

        logger.debug(f"Final Loss: {f(A, B):.6f}")

        if self.use_projection:
            # Go back to the original space
            self.Y = (U @ (Y @ B @ A).detach().cpu() + meanY).numpy()
            self.E = (U @ (Y @ B).detach().cpu() + meanY).numpy()
        else:
            self.Y = (Y @ B @ A).detach().cpu().numpy()
            self.E = (Y @ B).detach().cpu().numpy()
        self.A = A.detach().cpu().numpy()
        self.Xmap = B.t().detach().cpu().numpy()

        return self.E, self.A

    # @staticmethod
    # def update(a, b):
    #     m, _ = torch.max(b, dim=0, keepdim=True)
    #     num = a * torch.exp(b - m)
    #     denom = torch.sum(num, dim=0, keepdim=True)
    #     return num / denom

    @staticmethod
    def update(a, b):
        return F.softmax(torch.log(a) + b, dim=0)

    @staticmethod
    def get_steps_from_scheme(eta0: float, scheme: str, K: int):
        if scheme == "sqrt-simple":
            etas = eta0 / torch.sqrt(
                torch.arange(
                    start=1,
                    end=K + 1,
                )
            )
        elif scheme == "flat":
            etas = eta0 * torch.ones(K)
        else:
            raise NotImplementedError

        return etas


def check_f():
    L, N, p = 100, 10000, 6
    Y = torch.rand(L, N)
    B = torch.rand(N, p)
    A = torch.rand(p, N)

    res = 0.5 * ((Y - Y @ B @ A) ** 2).sum()

    assert res >= 0

    print(f"f(A, B, Y): {res:.4f}")


def check_grad_A():
    L, N, p = 100, 10000, 6
    Y = torch.rand(L, N)
    B = torch.rand(N, p)
    A = torch.rand(p, N)

    res = -B.t() @ Y.t() @ (Y - Y @ B @ A)

    assert res.shape == A.shape

    print(f"grad_A: {res}")


def check_grad_B():
    L, N, p = 100, 10000, 6
    Y = torch.rand(L, N)
    B = torch.rand(N, p)
    A = torch.rand(p, N)

    res = -Y.t() @ (Y - Y @ B @ A) @ A.t()

    assert res.shape == B.shape

    print(f"grad_B: {res}")


def check_alternatingEDA():
    params = {
        "eta0A": 0.02,
        "eta0B": 0.02,
        "K": 10,
        "nb_alternating": 4,
    }
    L, N, p = 100, 10000, 6
    Y = torch.rand(L, N)

    AEDA = AlternatingEDA(**params)

    E, A = AEDA.solve(Y, p)

    pdb.set_trace()
    # assert torch.all(E >= 0.0)
    # assert torch.all(E <= 1.0)
    # assert torch.allclose(A.)


if __name__ == "__main__":

    # check_f()
    # check_grad_A()
    # check_grad_B()

    check_alternatingEDA()

    # from hsi_unmixing.data.datasets.base import HSI
    # from hsi_unmixing.models.aligners import GreedyAligner as GA
    # from hsi_unmixing.models.initializers import VCA, TrueEndmembers

    # hsi = HSI("JasperRidge.mat")

    # params = {
    #     "K": 1000,
    #     "alpha_init": "softmax",
    #     "steps": "sqrt-simple",
    #     "eta0": 0.1,
    # }

    # vca = VCA()
    # te = TrueEndmembers()
    # # Einit = vca.init_like(hsi)
    # Einit = te.init_like(hsi)
    # solver = EDA(**params)
    # Yt, Et, At = hsi(asTensor=True)
    # Y0, E0, A0 = solver.solve(
    #     Yt,
    #     hsi.p,
    #     E0=torch.Tensor(Einit),
    # )

    # E0 = E0.detach().numpy()
    # A0 = A0.detach().numpy()

    # aligner = GA(hsi, "MeanAbsoluteError")
    # Ehat = aligner.fit_transform(E0)
    # Ahat = aligner.transform_abundances(A0)

    # hsi.plot_abundances(transpose=True)
    # hsi.plot_abundances(transpose=True, A0=Ahat)

    # hsi.plot_endmembers()
    # hsi.plot_endmembers(E0=Ehat)
