import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EDA(nn.Module):
    def __init__(
        self,
        n_bands,
        n_endmembers,
        unrollings=12,
        init_fn="softmax",
        init_eta=1.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers
        self.unrollings = unrollings
        self.init_fn = init_fn

        self.D = nn.Linear(
            self.n_endmembers,
            self.n_bands,
            bias=False,
        )

        # Endmember matrix initialization
        self.D.weight.data = torch.rand(self.n_bands, self.n_endmembers)

        self.etas = nn.ParameterList(
            [
                nn.Parameter(
                    # init_eta * torch.ones(1) / (k + 1),
                    init_eta
                    * torch.ones(1)
                )
                for k in range(self.unrollings)
            ],
        )

    def forward(self, x):

        # Initialization
        D = self.D.weight
        alpha = F.softmax(F.linear(x, D.t()), dim=1)

        print(f"Loss: {self.f(alpha, x):.4f} [0]")

        # Encoding
        for kk in range(self.unrollings):
            alpha = self.update(
                alpha,
                -self.etas[kk] * self.grad_f(alpha, x),
            )

            print(f"Loss: {self.f(alpha, x):.4f} [{kk+1}]")

        # Decoding
        x_hat = self.D(alpha)

        return x_hat, alpha

    def f(self, alpha, x):
        return 0.5 * torch.mean((x - self.D(alpha)) ** 2)

    def grad_f(self, alpha, x):
        D = self.D.weight
        return -F.linear(x - self.D(alpha), D.t())

    @staticmethod
    def update(a, b):
        m, _ = torch.max(b, dim=1, keepdim=True)
        return (a * torch.exp(b - m)) / torch.sum(
            a * torch.exp(b - m),
            dim=1,
            keepdim=True,
        )


def check_update():

    print("-" * 16)
    print("Checking Update")
    print("-" * 16)

    N, L, p, K = 4, 224, 3, 50

    x = torch.rand(N, L)

    init_eta = 1

    model = EDA(
        L,
        p,
        K,
        init_eta=init_eta,
    )

    # First iteration to be on the simplex
    alpha = F.softmax(F.linear(x, model.D.weight.t()), dim=1)

    print(f"Alpha:\n{alpha}")

    a = alpha
    b = model.grad_f(alpha, x)
    res = model.update(a, -b)

    print(f"Result:\n{res}")


def check_EDA():

    print("-" * 16)
    print("Checking EDA")
    print("-" * 16)

    N, L, p, K = 16, 224, 6, 50

    x = torch.rand(N, L)

    init_eta = 0.1

    model = EDA(
        L,
        p,
        K,
        init_eta=init_eta,
    )

    x_hat, abund = model(x)

    assert x_hat.shape == x.shape
    assert abund.shape == (N, p)


if __name__ == "__main__":
    check_EDA()
    check_update()
