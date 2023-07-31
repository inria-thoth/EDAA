import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MiSiCNet(nn.Module):
    def __init__(
        self,
        L=224,
        p=6,
        H=100,
        W=10,
        niters=8000,
        lr=0.001,
        exp_weight=0.99,
        lambd=100.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.L = L  # number of channels
        self.p = p  # number of endmembers
        self.H = H  # number of lines
        self.W = W  # number of samples per line

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.kernel_sizes = [3, 3, 3, 3, 1]
        self.strides = [1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.lambd = lambd

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        # MiSiCNet-like architecture
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[0]),
            nn.Conv2d(self.L, 256, self.kernel_sizes[0], stride=self.strides[0]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        # self.layer2 = nn.Sequential(
        #     nn.ReflectionPad2d(self.padding[1]),
        #     nn.Conv2d(256, 256, self.kernel_sizes[1], stride=self.strides[1]),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(**self.lrelu_params),
        # )

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.L, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            # nn.BatchNorm2d(260),
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, self.p, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(self.p),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.layer1(x)
        xskip = self.layerskip(x)
        xcat = torch.cat([x1, xskip], dim=1)
        abund = self.softmax(self.layer4(self.layer3(xcat)))
        abund_reshape = torch.transpose(abund.squeeze().view(-1, self.H * self.W), 0, 1)
        img = self.decoder(abund_reshape)
        return abund_reshape, img

    def loss(self, target, output):
        N, L = output.shape

        target_reshape = target.squeeze().reshape(L, N)
        fit_term = 0.5 * torch.linalg.norm(target_reshape.t() - output, "fro") ** 2

        O = target_reshape.mean(1, keepdims=True)
        reg_term = torch.linalg.norm(self.decoder.weight - O, "fro") ** 2

        return fit_term + self.lambd * reg_term

    @staticmethod
    def svd_projection(Y, p):
        V, SS, U = np.linalg.svd(Y, full_matrices=False)
        PC = np.diag(SS) @ U
        denoised_image_reshape = V[:, :p] @ PC[:p]
        return np.clip(denoised_image_reshape, 0, 1)

    def solve(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        self.init_architecture(seed=seed)

        Y = self.svd_projection(Y, p)
        # Initialize endmembers using SiVM extractor
        extractor = SiVM()
        Ehat = extractor.extract_endmembers(
            Y,
            p,
            seed=seed,
        )
        self.decoder.weight.data = torch.Tensor(Ehat)

        l, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, l, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)

        noisy_input = torch.rand_like(Y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.niters))
        for ii in progress:
            optimizer.zero_grad()

            abund, output = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * self.exp_weight + abund.detach() * (
                    1 - self.exp_weight
                )

            # Reshape data
            loss = self.loss(Y, output)

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            loss.backward()
            optimizer.step()
            # Enforce physical constraints on endmembers
            self.decoder.weight.data[self.decoder.weight <= 0] = 0
            self.decoder.weight.data[self.decoder.weight >= 1] = 1

        Ahat = out_avg.cpu().T.numpy()
        Ehat = self.decoder.weight.detach().cpu().numpy()
        self.time = time.time() - tic
        logger.info(f"MiSiCNet took {self.time:.2f}s")

        return Ehat, Ahat


class SiVM:
    def __init__(self):
        pass

    @staticmethod
    def Eucli_dist(x, y):
        a = np.subtract(x, y)
        return np.dot(a.T, a)

    def extract_endmembers(self, Y, p, seed=0, *args, **kwargs):

        x, p = Y, p

        [D, N] = x.shape
        # If no distf given, use Euclidean distance function
        Z1 = np.zeros((1, 1))
        O1 = np.ones((1, 1))
        # Find farthest point
        d = np.zeros((p, N))
        I = np.zeros((p, 1))
        V = np.zeros((1, N))
        ZD = np.zeros((D, 1))
        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), ZD)

        I = np.argmax(d[0, :])

        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))

        for v in range(1, p):
            D1 = np.concatenate(
                (d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1
            )
            D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
            D4 = np.concatenate((D1, D2), axis=0)
            D4 = np.linalg.inv(D4)

            for i in range(N):
                D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
                V[0, i] = np.dot(np.dot(D3.T, D4), D3)

            I = np.append(I, np.argmax(V))
            for i in range(N):
                d[v, i] = self.Eucli_dist(
                    x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1)
                )

        per = np.argsort(I)
        I = np.sort(I)
        d = d[per, :]
        E = x[:, I]
        logger.debug(f"Indices chosen: {I}")
        return E


if __name__ == "__main__":
    pass
