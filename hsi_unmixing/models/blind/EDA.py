import logging
import time

import numpy as np
import scipy.sparse as sp
import spams
import torch
import torch.nn.functional as F
import wandb
from hsi_unmixing.models.initializers import VCA
from hsi_unmixing.models.metrics import SADDegrees, aRMSE
from sklearn.cluster import KMeans
from tqdm import tqdm

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
        B_init="random",
        schemeA="sqrt-simple",
        schemeB="sqrt-simple",
        nb_alternating=10,
        device=None,
        use_projection=False,
        entropic_regularization=False,
        denoise=False,
        epsilon=1e-3,
        log_every_n_steps=10,
        coef=1.1,
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
        self.B_init = B_init
        self.etasA = self.get_steps_from_scheme(eta0A, schemeA, self.KA)
        self.etasB = self.get_steps_from_scheme(eta0B, schemeB, self.KB)
        self.nb_alternating = nb_alternating
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.use_projection = use_projection
        self.denoise = denoise
        self.entropic_regularization = entropic_regularization
        self.epsilon = epsilon
        self.log_every_n_steps = log_every_n_steps
        self.coef = coef
        self.eta0A = eta0A
        self.eta0B = eta0B
        cfg = {
            "KA": self.KA,
            "KB": self.KB,
            "eta0A": self.eta0A,
            "eta0B": self.eta0B,
            "eps": self.epsilon,
            "nb_alternating": self.nb_alternating,
            "coef": self.coef,
        }
        self.runner = wandb.init(
            project="HSU",
            config=cfg,
            # job_type="hparams",
            # job_type="Urban4Radius|epsilon|fixed_steps",
            job_type="Urban4Radius",
            name=f"eps{epsilon}_A{eta0A}_B{eta0B}_c{coef}",
            # name=f"eps{epsilon}",
            notes=f"KA{KA}_KB{KB}_eta0A{eta0A}_eta0B{eta0B}_eps{epsilon}_Kglob{nb_alternating}",
        )

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def solve(
        self,
        Y,
        p,
        seed,
        hsi,
        E0=None,
        aligner=None,
        tol=1e-45,
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

        # Noise variance estimation
        sigma = self.estAdditiveNoise(Y.detach().numpy())
        # threshold = 1.1 * N * L * sigma
        threshold = self.coef * N * L * sigma
        logger.info(f"Estimated sigma: {sigma:.6f}")
        logger.info(f"Estimated threshold: {threshold:.6f}")

        # Fix seed
        torch.manual_seed(seed)

        # YtY = Y.t() @ Y
        # Id = torch.eye(N)
        if self.denoise:
            logger.debug(f"Denoise data using SVD")
            U = torch.linalg.svd(Y, full_matrices=False)[0][:, :p]
            Y = U @ U.t() @ Y

        # Compute projection to (p - 1)-subspace here
        if self.use_projection:
            logger.debug(f"Y shape before projection: {Y.shape}")
            # center Y
            meanY = Y.mean(1, keepdims=True)
            Y -= meanY
            diagY = Y @ Y.t() / N
            # U = torch.linalg.svd(diagY, full_matrices=False)[0][:, : p - 1]
            U = torch.linalg.svd(diagY, full_matrices=False)[0][:, :p]
            Y = U.t() @ Y
            logger.debug(f"Y shape after projection: {Y.shape}")

        # Inner functions
        rmse = aRMSE()
        sad = SADDegrees()

        E_gt = hsi.E
        A_gt = hsi.A

        def residual(a, b):
            return 0.5 * ((Y - Y @ b @ a) ** 2).sum()

        def entropy(x):
            # return -(x * (torch.log(x) - 1)).sum()
            ret = torch.where(
                x > tol,
                -(x * (torch.log(x) - 1)).to(torch.float64),
                0.0,
            ).sum()
            return ret

        # def f(a, b):
        #     fit_term = 0.5 * ((Y - Y @ b @ a) ** 2).sum()
        #     if self.entropic_regularization:
        #         # entropy = self.epsilon * (a * (torch.log(a) - 1)).sum()
        #         entropy = (a * (torch.log(a) - 1)).sum()
        #         logger.debug(f"\tEntropy: {entropy:.6f}")
        #         return fit_term - self.epsilon * entropy
        #     else:
        #         return fit_term

        if self.entropic_regularization:

            def loss(a, b):
                return residual(a, b) + self.epsilon * entropy(a)

        else:

            def loss(a, b):
                return residual(a, b)

        def grad_A(a, b):
            fit_term = -b.t() @ Y.t() @ (Y - Y @ b @ a)
            if self.entropic_regularization:
                # return fit_term + self.epsilon * (torch.log(a) + ones_pN)
                # return fit_term - self.epsilon * (torch.log(a) - 1)
                return fit_term - self.epsilon * (torch.log(a) + 1)
                # return fit_term + self.epsilon * torch.log(a)
            else:
                return fit_term
            # return -b.t() @ YtY @ (Id - b @ a)

        def grad_B(a, b):
            # return -Y.t() @ Y @ a.t() + Y.t() @ Y @ b @ a @ a.t()
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())
            # return -YtY @ (Id - b @ a) @ a.t()

        # Initialization
        # B = F.softmax(torch.ones(N, p) + 0.01 * torch.randn(N, p), dim=0)
        if self.B_init == "randn":
            B = F.softmax(torch.randn(N, p), dim=0)
        elif self.B_init == "rand":
            B = F.softmax(torch.rand(N, p), dim=0)
        elif self.B_init == "uniform":
            B = torch.ones(N, p) / N
        elif self.B_init == "indices":
            indices = torch.randint(0, high=N - 1, size=(p,))
            E = Y[:, indices]
            # B = F.softmax(torch.linalg.solve(Y, E), dim=0)
            B = F.softmax(torch.linalg.pinv(Y) @ E, dim=0)
        elif self.B_init == "init":
            if E0 is not None:
                E0 = torch.Tensor(E0)
                if self.use_projection:
                    E0 = U.t() @ E0
            # B = F.softmax(torch.linalg.solve(Y, E0), dim=0)
            B = F.softmax(torch.linalg.pinv(Y) @ E0, dim=0)
        elif self.B_init == "pSVD":
            _, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            Vhp = Vh[:p]
            S_inv = torch.linalg.inv(torch.diag(S[:p]))
            B0 = Vhp.t() @ S_inv
            B = F.softmax(B0, dim=0)
        elif self.B_init == "pSVD_l1":
            _, S, Vh = torch.linalg.svd(Y, full_matrices=False)

            Vhp = Vh[:p]
            S_inv = torch.linalg.inv(torch.diag(S[:p]))
            B0 = Vhp.t() @ S_inv
            # L1 projection
            B = torch.abs(B0) / torch.sum(torch.abs(B0), dim=0, keepdims=True)
        else:
            raise NotImplementedError
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
        elif self.A_init == "random":
            A = F.softmax(torch.randn(p, N), dim=0)
        else:
            raise NotImplementedError

        # print(f"Loss: {f(A, B):.6f} [0]")

        # To device
        Y = Y.to(self.device)
        A = A.to(self.device)
        B = B.to(self.device)
        # ones_pN = torch.ones(p, N, device=self.device)

        etasA = self.etasA.to(self.device)
        etasB = self.etasB.to(self.device)

        initial_loss_value = round(loss(A, B).item(), 2)
        initial_residual_value = round(residual(A, B).item(), 2)
        initial_entropy_value = round(entropy(A).item(), 2)
        logger.debug(f"Initial loss: {initial_loss_value}")
        E0 = (Y @ B).detach().cpu().numpy()
        E1 = aligner.fit_transform(E0)
        A0 = A.detach().cpu().numpy()
        A1 = aligner.transform_abundances(A0)
        sad_value = round(sad(E1, E_gt), 2)
        rmse_value = round(rmse(A1, A_gt), 2)
        self.runner.log(
            {
                "loss": initial_loss_value,
                "RMSE": rmse_value,
                "SAD": sad_value,
                "residual": initial_residual_value,
                "entropy": initial_entropy_value,
            }
        )
        # etasA = 1.0 / p
        # etasB = 1.0 / N
        etasA = self.eta0A
        etasB = self.eta0B

        # symA = B.t() @ Y.t() @ Y @ B
        # # eigvalA_max = torch.linalg.eigvalsh(symA)[-1]
        # # logger.debug(f"Max. eigval for symA => {eigvalA_max}")
        # # etasA = 1.0 / eigvalA_max
        # sA_max = torch.linalg.matrix_norm(symA, ord=2)
        # logger.debug(f"Max. singular value for A => {sA_max}")
        # etasA = 1.0 / sA_max
        logger.debug(f"etasA: {etasA}")

        # symB0 = A @ Y.t() @ Y @ A.t()
        # symB1 = A @ A.t()
        # # symB2 = Y @ Y.t()
        # # # eigvalB0_max = torch.linalg.eigvalsh(symB0)[-1]
        # # eigvalB1_max = torch.linalg.eigvalsh(symB1)[-1]
        # # eigvalB2_max = torch.linalg.eigvalsh(symB2)[-1]
        # # # logger.debug(f"Max. eigval for symB0 => {eigvalB0_max}")
        # # logger.debug(f"Max. eigval for symB1 => {eigvalB1_max}")
        # # logger.debug(f"Max. eigval for symB2 => {eigvalB2_max}")
        # # # etasB = 1.0 / min(eigvalB0_max, eigvalB1_max, eigvalB2_max)
        # # etasB = 1.0 / min(eigvalB1_max, eigvalB2_max)

        # sB_max = torch.linalg.matrix_norm(symB1, ord=2)
        # logger.debug(f"Max singular value for B => {sB_max}")
        # etasB = 1.0 / sB_max
        logger.debug(f"etasB: {etasB}")

        # breakpoint()

        # eta = etasB[0]
        iters = 0
        with torch.no_grad():
            # Encoding
            while residual(A, B).item() > threshold:
                # for ii in range(self.nb_alternating):
                # if ii % 100 == 50:
                #     MA = B.t() @ Y.t() @ Y @ B
                #     MB = Y.t() @ Y @ A.t() @ A
                #     etasA = self.compute_etas(
                #         M=MA,
                #         dim=p,
                #         K=self.KA,
                #         device=self.device,
                #     )
                #     etasB = self.compute_etas(
                #         M=MB,
                #         dim=N,
                #         K=self.KB,
                #         device=self.device,
                #     )
                if iters % 2 == 0:
                    # if ii % 2 == 0:
                    for kk in range(self.KA):
                        A = self.update(
                            A,
                            # -self.etasA[kk] / (ii + 1) * grad_A(A, B),
                            # -self.etasA[kk] * grad_A(A, B),
                            # -etasA[kk] * grad_A(A, B),
                            -etasA * grad_A(A, B),
                        )
                        # if kk == 0:
                        #     pass
                        if kk % self.log_every_n_steps == 0:
                            loss_value = round(loss(A, B).item(), 2)
                            residual_value = round(residual(A, B).item(), 2)
                            entropy_value = round(entropy(A).item(), 2)
                            # logger.debug(f"Loss: {loss_value} [{ii}|{kk+1}]")
                            logger.debug(f"Loss: {loss_value} [{iters}|{kk+1}]")
                            E0 = (Y @ B).detach().cpu().numpy()
                            E1 = aligner.fit_transform(E0)
                            A0 = A.detach().cpu().numpy()
                            A1 = aligner.transform_abundances(A0)
                            sad_value = round(sad(E1, E_gt), 2)
                            rmse_value = round(rmse(A1, A_gt), 2)
                            self.runner.log(
                                {
                                    "loss": loss_value,
                                    "RMSE": rmse_value,
                                    "SAD": sad_value,
                                    "residual": residual_value,
                                    "entropy": entropy_value,
                                }
                            )

                        # print(f"Loss: {f(A, B):.6f} [{ii}|{kk+1}]")
                    iters += 1

                else:
                    for kk in range(self.KB):
                        B = self.update(
                            B,
                            # -self.etasB[kk] / ii * grad_B(A, B),
                            # -self.etasB[kk] * grad_B(A, B),
                            # -etasB[kk] * grad_B(A, B),
                            -etasB * grad_B(A, B),
                        )
                        if kk % self.log_every_n_steps == 0:
                            loss_value = round(loss(A, B).item(), 2)
                            residual_value = round(residual(A, B).item(), 2)
                            entropy_value = round(entropy(A).item(), 2)
                            E0 = (Y @ B).detach().cpu().numpy()
                            E1 = aligner.fit_transform(E0)
                            A0 = A.detach().cpu().numpy()
                            A1 = aligner.transform_abundances(A0)
                            sad_value = round(sad(E1, E_gt), 2)
                            rmse_value = round(rmse(A1, A_gt), 2)
                            # logger.debug(f"Loss: {loss_value} [{ii}|{kk+1}]")
                            logger.debug(f"Loss: {loss_value} [{iters}|{kk+1}]")
                            self.runner.log(
                                {
                                    "loss": loss_value,
                                    "RMSE": rmse_value,
                                    "SAD": sad_value,
                                    "residual": residual_value,
                                    "entropy": entropy_value,
                                }
                            )

                    iters += 1

        tac = time.time()
        self.time = round(tac - tic, 2)
        logger.info(f"{self} took {self.time:.2f}s")

        loss_value = round(float(loss(A, B)), 2)
        logger.debug(f"Final Loss: {loss_value}")
        self.runner.log({"loss": loss_value})

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
    def estAdditiveNoise(r):
        small = 1e-6
        L, N = r.shape
        w = np.zeros((L, N))
        RR = np.dot(r, r.T)
        RRi = np.linalg.pinv(RR + small * np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:, i] * RRi[i, :]) / RRi[i, i]
            RRa = RR[:, i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0, i] = 0
            w[i, :] = r[i, :] - np.dot(beta, r)
        Rw = np.diag(np.diag(np.dot(w, w.T) / N))
        # breakpoint()
        logger.debug(f"Sigma diagonal: {Rw.diagonal()}")
        sigma = Rw.diagonal().mean()
        return sigma

    @staticmethod
    def update(a, b):
        return F.softmax(torch.log(a) + b, dim=0)

    @staticmethod
    def get_steps_from_scheme(
        eta0: float,
        scheme: str,
        K: int,
    ):
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

    @staticmethod
    def compute_etas(M, dim, K, device):
        logger.debug("Computing eta...")
        num = torch.sqrt(2 * torch.log(torch.Tensor([dim]))).to(device)
        Lf = torch.linalg.svdvals(M)[0]
        logger.debug(f"Lf: {Lf:.3f}")
        denom = Lf * torch.sqrt(
            torch.arange(
                start=1,
                end=K + 1,
            ).to(device)
        )
        ret = num / denom
        logger.debug(f"First 10 steps: {ret[:10]}")
        return ret


class DSEDA:
    def __init__(
        self,
        T=100,
        Ka=50,
        Kb=2,
        epsilon=1e-3,
        Binit="soft",
        Ainit="soft",
        centering=False,
        # entropic_reg=True,
        # log_every_n_steps=10,
    ):
        self.T = T
        self.Ka = Ka
        self.Kb = Kb
        self.epsilon = epsilon
        # self.log_every_n_steps = log_every_n_steps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Binit = Binit
        self.Ainit = Ainit
        self.centering = centering

    def solve(
        self,
        Y,
        p,
        hsi,
        seed,
        aligner=None,
        tol=1e-40,
        timesteps=[20, 50, None],
        runner_on=False,
        mode="blind",
        **kwargs,
    ):

        if runner_on:
            runner = wandb.init(
                project="HSU",
                name=f"{hsi.shortname}_{self}_eps{self.epsilon}_Ka{self.Ka}_Kb{self.Kb}",
            )

        # Fix seed
        torch.manual_seed(seed)

        # Metrics
        rmse = aRMSE()
        sad = SADDegrees()

        E_gt = hsi.E
        A_gt = hsi.A

        results = {}

        tic = time.time()

        L, N = Y.shape

        def residual(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def entropy(x):
            ret = torch.where(
                x > tol,
                -(x * (torch.log(x))).to(torch.float64),
                0.0,
            ).sum()
            return ret

        def loss(a, b):
            return residual(a, b) + self.epsilon * entropy(a)

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b, epsilon):
            fact = 1 / (1 - epsilon * self.etaA)
            return F.softmax(fact * torch.log(a) + b, dim=0)

        def computeLA(a, b, mY):
            # breakpoint()
            YB = (Y + mY) @ b
            # YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        def update_spamsB(a, b):
            # Update one column b
            # |R - Y b a.T|^2
            # |R|^2 - 2 < Y.T R a, b > + a.T a < Y b  , Y b >
            # |R|^2 - 2 < R a, Y b > + a.T a < Y b  , Y b >
            # | R a/ |a|^2  - Yb |^2
            R = Y - (Y @ b) @ a
            YY = np.asfortranarray(Y.cpu().numpy().astype(float))
            for ii in range(p):
                R = R + (Y @ b[:, ii]).unsqueeze(1) @ a[ii, :].unsqueeze(0)
                z = (R @ a[ii, :]) / (a[ii, :] @ a[ii, :].t())
                z = np.asfortranarray(z.unsqueeze(1).cpu().numpy().astype(float))
                bb = spams.decompSimplex(z, YY)
                b[:, ii] = torch.tensor(bb.todense().astype(np.float32)).squeeze()
                R = R - (Y @ b[:, ii]).unsqueeze(1) @ a[ii, :].unsqueeze(0)
            return b

        with torch.no_grad():
            # B init
            # B0 = torch.randn((N, p))

            meanY = Y.mean(1, keepdims=True) if self.centering else 0

            # breakpoint()
            Y -= meanY

            B, A = kmeans_init(Y, p)

            # A = A + 0.01
            # A /= A.sum(0, keepdims=True)
            # B = B + 0.01 * B.max()
            # B /= B.sum(0, keepdims=True)
            _, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            Vhp = Vh[:p]
            S_inv = torch.linalg.inv(torch.diag(S[:p]))
            B0 = Vhp.t() @ S_inv
            if self.Binit == "soft":
                B = F.softmax(B0, dim=0)
            elif self.Binit == "proj":
                # L1 projection
                B = torch.abs(B0) / torch.sum(
                    torch.abs(B0),
                    dim=0,
                    keepdims=True,
                )
            elif self.Binit == "rand":
                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                # B = F.softmax(torch.rand((N, p)), dim=0)
            elif self.Binit == "uniform":
                B = F.softmax(
                    (1 / N) * torch.ones_like(B) + 0.001 * torch.rand_like(B), dim=0
                )
            elif self.Binit == "kmeans":
                pass
            else:
                raise NotImplementedError

            # A init
            if self.Ainit == "soft":
                # A = torch.randn((p, N))
                A = F.softmax(-grad_A(A, B), dim=0)
            elif self.Ainit == "DS":
                # Decomp Simplex
                Yf = np.asfortranarray(Y.numpy(), dtype=np.float64)
                Ef = np.asfortranarray((Y @ B).numpy(), dtype=np.float64)

                W = spams.decompSimplex(
                    Yf,
                    Ef,
                    computeXtX=True,
                    numThreads=-1,
                )
                A = torch.Tensor(sp.csr_matrix.toarray(W))
            elif self.Ainit == "deterministic":
                A = F.softmax(B.t() @ Y.t() @ Y, dim=0)
            elif self.Ainit == "Bt":
                A = F.softmax(B.t(), dim=0)
            elif self.Ainit == "uniform":
                A = (1 / p) * torch.ones_like(A)
            elif self.Ainit == "kmeans":
                pass
            else:
                raise NotImplementedError

            # Device loading
            Y = Y.to(self.device)
            meanY = meanY.to(self.device) if self.centering else 0

            # meanY = 0
            A = A.to(self.device)
            B = B.to(self.device)
            best_res_c = np.inf

            self.etaA = 1.0 / computeLA(A, B, meanY)
            # self.etaB = self.etaA * p / N
            # self.etaB = B.max()
            self.etaB = self.etaA * ((p / N) ** 0.5)
            # self.etaA = 1.0
            # self.etaA = 0.5
            # self.etaA = 0.1
            # self.etaA = 0.27
            print(f"eta A value: {self.etaA}")
            print(f"eta B value: {self.etaB}")
            # breakpoint()

            # Main loop
            for ii in tqdm(range(self.T)):
                for kk in range(self.Ka):
                    A = update(A, -self.etaA * grad_A(A, B), self.epsilon)
                # if kk % self.log_every_n_steps == 0:
                #     loss_value = round(loss(A, B).item(), 2)
                #     logger.debug(f"Loss: {loss_value} [{ii}|A:{kk+1}]")
                # print(A[0])

                for kk in range(self.Kb):
                    # B = update_spamsB(A, B)
                    B = update(B, -self.etaB * grad_B(A, B), 0.0)
                    # if kk % self.log_every_n_steps == 0:
                    # loss_value = round(loss(A, B).item(), 2)
                    # logger.debug(f"Loss: {loss_value} [{ii}|B:{kk+1}]")

                if (ii + 1) in timesteps:
                    results[ii + 1] = {
                        "E": ((Y + meanY) @ B).cpu().numpy(),
                        "A": A.cpu().numpy(),
                    }

                # # if ii % 10 == 9:
                # res_c = inpainting(
                #     Y.cpu().numpy(),
                #     B.cpu().numpy(),
                # )

                # if res_c < best_res_c:
                #     best_res_c = res_c

                #     self.Y = (Y @ B @ A).cpu().numpy()
                #     self.E = ((Y + meanY) @ B).cpu().numpy()
                #     self.A = A.cpu().numpy()
                #     self.Xmap = B.t().cpu().numpy()

                loss_value = round(loss(A, B).item(), 2)
                curr_residual = residual(A, B).item()
                residual_value = round(curr_residual, 2)
                curr_residual_normed = 2 * curr_residual / (Y ** 2).sum()
                residual_normed_value = round(curr_residual_normed.item(), 2)
                entropy_value = round(entropy(A).item(), 2)
                # inpainting_residual = round(res_c, 2)
                sparsity = round(self.compute_sparsity(A).item(), 2)

                E0 = ((Y + meanY) @ B).cpu().numpy()
                # E1 = aligner.fit_transform(E0)
                A0 = A.cpu().numpy()
                # A1 = aligner.transform_abundances(A0)
                A1 = aligner.fit_transform(A0)
                E1 = aligner.transform_endmembers(E0)
                sad_value = round(sad(E1, E_gt), 2)
                rmse_value = round(rmse(A1, A_gt), 2)

                if runner_on:
                    runner.log(
                        {
                            "loss": loss_value,
                            "residual": residual_value,
                            "normed_residual": curr_residual_normed,
                            "entropy": entropy_value,
                            "aRMSE": rmse_value,
                            "SAD": sad_value,
                            # "inpainting_residual": inpainting_residual,
                            "sparsity": sparsity,
                        }
                    )

            # if None in timesteps:
            #     results["inpainting"] = {
            #         "E": self.E,
            #         "A": self.A,
            #     }

        exc_time = round(time.time() - tic, 2)
        logger.info(f"{self} took {exc_time} seconds")

        self.Y = ((Y + meanY) @ B @ A).cpu().numpy()
        self.E = ((Y + meanY) @ B).cpu().numpy()
        self.A = A.cpu().numpy()
        self.Xmap = B.t().cpu().numpy()

        # return self.E, self.A
        if mode == "multistop":
            return results
        elif mode == "blind":
            return self.E, self.A

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    @staticmethod
    def compute_sparsity(A, tol=0.01):
        return (A <= tol).sum() / A.numel()


class EDAv1:
    def __init__(
        self,
        T=100,
        Ka=100,
        Kb=100,
        epsilon=1e-3,
    ):
        self.T = T
        self.Ka = Ka
        self.Kb = Kb
        self.epsilon = epsilon

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.endmembers = []

    def solve(
        self,
        Y,
        p,
        hsi,
        runs,
        tol=1e-40,
        **kwargs,
    ):
        best_E = None
        best_A = None
        min_max_corrcoef = 10.0
        max_det = 0.0
        min_SSD = 1e10

        L, N = Y.shape

        def residual(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def entropy(x):
            ret = torch.where(
                x > tol,
                -(x * (torch.log(x))).to(torch.float64),
                0.0,
            ).sum()
            return ret

        def loss(a, b):
            return residual(a, b) + self.epsilon * entropy(a)

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b, epsilon):
            fact = 1 / (1 - epsilon * self.etaA)
            return F.softmax(fact * torch.log(a) + b, dim=0)

        def computeLA(a, b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        def SSD(e):
            ret = 0
            for ll in range(L):
                ret += (
                    e[ll][None, :] @ (p * np.eye(p) - np.ones((p, p))) @ e[ll][:, None]
                )
            return np.asscalar(ret)

        max_correl = lambda e: np.max(np.corrcoef(e.T) - np.eye(p))
        det = lambda e: np.abs(np.linalg.det(e.T @ e))

        tic = time.time()

        for run in tqdm(range(runs)):
            torch.manual_seed(run)

            with torch.no_grad():

                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                # B = F.softmax(0.01 * torch.rand((N, p)), dim=0)
                # B = VCA_init(hsi=hsi, seed=run)
                A = (1 / p) * torch.ones((p, N))

                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)

                for ii in range(self.T):
                    if ii % self.T == 0:
                        # if ii % 10 == 0:
                        self.etaA = 1.0 / computeLA(A, B)
                        # self.etaB = self.etaA * ((p / N) ** 0.5)
                        self.etaB = self.etaA * (p / N)
                    for kk in range(self.Ka):
                        A = update(A, -self.etaA * grad_A(A, B), self.epsilon)

                    for kk in range(self.Kb):
                        B = update(B, -self.etaB * grad_B(A, B), 0.0)

                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                non_diagonal_max_corrcoef = max_correl(E)
                curr_det = det(E)
                ssd_value = SSD(E)
                logger.info(f"Current det => {round(curr_det, 5)}")
                logger.info(f"NDMCC => {round(non_diagonal_max_corrcoef, 3)}")
                logger.info(f"SSD => {round(SSD(E), 3)}")
                # if curr_det > max_det:
                if non_diagonal_max_corrcoef < min_max_corrcoef:
                    # if ssd_value < min_SSD:
                    min_max_corrcoef = non_diagonal_max_corrcoef
                    # min_SSD = ssd_value
                    logger.info("MIN!")
                    # logger.info("MAX!")
                    # max_det = curr_det
                    best_E = E
                    best_A = A
                    self.Xmap = B.t().cpu().numpy()

                    self.endmembers.append(E)

        toc = time.time()
        elapsed_time = round(toc - tic, 2)
        logger.info(f"{self} took {elapsed_time}s")
        logger.info(f"{len(self.endmembers)} bundles selected...")

        return best_E, best_A

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg


def VCA_init(hsi, seed):
    vca = VCA()
    _ = vca.init_like(hsi, seed)
    indices = vca.indices

    N, p = hsi.N, hsi.p

    B = np.zeros((N, p))
    for pp, ii in enumerate(indices):
        B[ii, pp] = 1.0

    assert np.allclose(B.sum(0), np.ones(p))
    B = torch.Tensor(B)
    return B


def kmeans_init(Y, p):
    X = Y.numpy().T

    N, L = X.shape

    # kmeans = KMeans(n_clusters=p, n_init=10)
    kmeans = KMeans(n_clusters=p, n_init=1)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    logger.info(np.unique(labels, return_counts=True))
    A = np.zeros((p, N))
    dists = np.zeros((N, p))
    # Cluster assignment
    for ii, label in enumerate(labels):
        A[label, ii] = 1.0
    for jj in range(N):
        dists[jj] = np.sum((X[jj] - centroids) ** 2, axis=1)
    # breakpoint()
    B = np.copy(A.T)
    # Weighted by distance to centroids
    B = 1 / dists
    # B = dists
    # Normalize B
    # B /= B.sum(0, keepdims=True)
    assert np.allclose(A.sum(0), np.ones(N))
    # assert np.allclose(B.sum(0), np.ones(p))
    A = torch.Tensor(A)
    B = torch.Tensor(B)
    B = F.softmax(B, dim=0)
    return B, A


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

    # assert torch.all(E >= 0.0)
    # assert torch.all(E <= 1.0)
    # assert torch.allclose(A.)


def inpainting(Y, B, mask=None):
    # Hardcode mask first
    L, N = Y.shape
    Z = Y @ B
    p = 0.97
    np.random.seed(42)
    mask = np.random.binomial(1, p, L)

    # (1 - p) x 100% channels are selected
    Y_M = Y[mask == 0]
    # print(f"Y_M shape => {Y_M.shape}")
    Y_Mc = Y[mask == 1]
    # print(f"Y_Mc shape => {Y_Mc.shape}")

    Z_M = Z[mask == 0]
    # print(f"Z_M shape => {Z_M.shape}")
    Z_Mc = Z[mask == 1]
    # print(f"Z_Mc shape => {Z_Mc.shape}")

    x = np.asfortranarray(Y_M).astype(float)
    z = np.asfortranarray(Z_M).astype(float)

    ds = spams.decompSimplex(x, z)
    a = np.asarray(ds.todense().astype(np.float32))
    residual = ((Y_M - Z_M @ a) ** 2).sum()
    print(f"Mask Residual => {residual:.4f}")

    residual_c = ((Y_Mc - Z_Mc @ a) ** 2).sum()
    print(f"Complementary Residual => {residual_c:.4f}")

    return residual_c


def plot_me(x):
    import matplotlib.pyplot as plt

    _, p = x.shape

    for ii in range(p):
        plt.plot(x[:, ii])

    plt.show()


def check_inpainting():
    L, N, p = 100, 1000, 5
    Y = np.random.rand(L, N)
    B = np.random.rand(N, p)

    inpainting(Y, B)


if __name__ == "__main__":

    check_inpainting()

    # check_f()
    # check_grad_A()
    # check_grad_B()

    # check_alternatingEDA()

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
