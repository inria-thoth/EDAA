import logging
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SupervisedEDAA:
    def __init__(self, K=1000, epsilon=0.0):
        self.K = K
        self.epsilon = epsilon
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def solve(
        self,
        Y,
        E,
        **kwargs,
    ):

        E = torch.Tensor(E)
        assert Y.shape[0] == E.shape[0]
        L, N = Y.shape
        _, p = E.shape

        def grad_A(a):
            return -E.t() @ (Y - E @ a)

        def update(a, b, epsilon):
            fact = 1 / (1 - epsilon * self.etaA)
            return F.softmax(fact * torch.log(a) + b, dim=0)

        def computeLA():
            S = torch.linalg.svdvals(E)
            return S[0] * S[0]

        with torch.no_grad():

            A = (1 / p) * torch.ones((p, N))

            Y = Y.to(self.device)
            E = E.to(self.device)
            A = A.to(self.device)

            self.etaA = 1.0 / computeLA()

            for kk in range(self.K):
                A = update(A, -self.etaA * grad_A(A), self.epsilon)

            A = A.cpu().numpy()

        return A


if __name__ == "__main__":
    pass
