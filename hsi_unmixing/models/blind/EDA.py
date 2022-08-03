import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BlindEDA:
    def __init__(self, T=50, K1=20, K2=20, M=100):
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.endmembers = []

    def solve(
        self,
        Y,
        p,
        seed=0,
        **kwargs,
    ):
        best_E = None
        best_A = None
        min_max_corrcoef = 10.0

        L, N = Y.shape

        def residual(a, b):
            return 0.5 * ((Y - (Y @ b) @ a) ** 2).sum()

        def residual_l1(a, b):
            return (Y - (Y @ b) @ a).abs().sum()

        def loss(a, b):
            return residual(a, b)

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b):
            return F.softmax(torch.log(a) + b, dim=0)

        def computeLA(a, b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        max_correl = lambda e: np.max(np.corrcoef(e.T) - np.eye(p))

        results = {}

        tic = time.time()

        for m in tqdm(range(self.M)):
            torch.manual_seed(m + seed)
            generator = np.random.RandomState(m + seed)

            with torch.no_grad():

                # Matrix initialization
                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                A = (1 / p) * torch.ones((p, N))

                # Send matrices on GPU
                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)

                # Random Step size factor
                factA = 2 ** generator.randint(-3, 4)

                # Compute step sizes
                self.etaA = factA / computeLA(A, B)
                self.etaB = self.etaA * ((p / N) ** 0.5)

                for ii in range(self.T):
                    for kk in range(self.K1):
                        A = update(A, -self.etaA * grad_A(A, B))

                    for kk in range(self.K2):
                        B = update(B, -self.etaB * grad_B(A, B))

                fit_m = residual_l1(A, B).item()
                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                Xmap = B.t().cpu().numpy()
                Rm = max_correl(E)
                # Store results
                results[m] = {
                    "Rm": Rm,
                    "Em": E,
                    "Am": A,
                    "Bm": Xmap,
                    "fit_m": fit_m,
                    "factA": factA,
                }

        min_fit_l1 = np.min([v["fit_m"] for k, v in results.items()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]["fit_m"]
            return (abs(val - min_fit_l1) / abs(val)) < tol

        sorted_indices = sorted(
            filter(fit_l1_cutoff, results),
            key=lambda x: results[x]["Rm"],
        )

        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]

        best_E = best_result["Em"]
        best_A = best_result["Am"]
        self.Xmap = best_result["Bm"]

        toc = time.time()
        elapsed_time = round(toc - tic, 2)
        logger.info(f"{self} took {elapsed_time}s")

        return best_E, best_A

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg


if __name__ == "__main__":
    pass
