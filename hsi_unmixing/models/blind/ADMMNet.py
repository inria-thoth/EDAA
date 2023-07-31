"""
ADMMNet simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import numpy.linalg as LA
import torch.nn as nn
import torch
import torch.nn.functional as F

EPS = 1e-6

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class X_block(nn.Module):
    def __init__(self, L, p, A_init, mu, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.W = nn.Linear(L, p, bias=False)
        self.B = nn.Linear(p, p, bias=False)

        # init
        M = A_init.T @ A_init + mu * torch.eye(p)

        self.W.weight.data = torch.linalg.solve(M, A_init.T)
        self.B.weight.data = torch.linalg.solve(M, mu * torch.eye(p))

    def forward(self, y, z, d):
        return self.W(y) + self.B(z + d)


class D_block(nn.Module):
    def __init__(self, eta_init=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eta = nn.Parameter(
            data=eta_init * torch.ones(1),
            requires_grad=True,
        )

    def forward(self, x, z, d):
        return d - self.eta * (x - z)


class Z_block(nn.Module):
    def __init__(self, p, theta_init=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # init
        self.theta = nn.Parameter(
            data=theta_init * torch.ones(p),
            requires_grad=True,
        )

    def forward(self, x, d):
        return F.relu(x - d - self.theta)


class ADMMNet(nn.Module):
    def __init__(
        self,
        lr,
        epochs,
        batchsize,
        nblocks,
        lambd,
        mu,
        tied,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.nblocks = nblocks
        self.tied = tied

        # Hyperparameters
        self.lambd = lambd
        self.mu = mu

    def init_architecture(
        self,
        A_init,
        eta_init=1.0,
    ):

        self.x_blocks = nn.ModuleList()
        self.z_blocks = nn.ModuleList()
        self.d_blocks = nn.ModuleList()

        # NOTE this is for tied params

        if self.tied:
            x_block = X_block(self.L, self.p, A_init=A_init, mu=self.mu)
            z_block = Z_block(self.p, theta_init=self.lambd / self.mu)
            d_block = D_block(eta_init=eta_init)

            for _ in range(self.nblocks):
                self.x_blocks.append(x_block)
                self.z_blocks.append(z_block)
                self.d_blocks.append(d_block)

        else:
            for _ in range(self.nblocks):
                self.x_blocks.append(X_block(self.L, self.p, A_init=A_init, mu=self.mu))
                self.z_blocks.append(Z_block(self.p, theta_init=self.lambd / self.mu))
                self.d_blocks.append(D_block(eta_init=eta_init))

        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.decoder.weight.data = A_init

    def forward(self, y):
        bs, l = y.shape
        z = torch.zeros((bs, self.p)).to(self.device)
        d = torch.zeros((bs, self.p)).to(self.device)
        for ii in range(self.nblocks):
            x = self.x_blocks[ii](y, z, d)
            z = self.z_blocks[ii](x, d)
            d = self.d_blocks[ii](x, z, d)

        abund = z
        abund = abund / (abund.sum(1, keepdims=True) + EPS)
        output = self.decoder(abund)
        return abund, output

    def solve(self, Y, p, *args, **kwargs):

        tic = time.time()
        logger.debug("Solving started...")

        L, N = Y.shape

        # Hyperparameters
        self.L = L
        self.p = p

        # endmembers initialization
        extractor = VCA()
        Ehat = extractor.extract_endmembers(Y, p)
        A_init = torch.Tensor(Ehat)
        self.init_architecture(A_init=A_init)

        self = self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_db = torch.utils.data.TensorDataset(torch.Tensor(Y.T))
        dataloader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batchsize,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        self.train()

        for ii in progress:
            running_loss = 0
            for x, y in enumerate(dataloader):
                y = y[0].to(self.device)

                abund, output = self(y)

                loss = F.mse_loss(y, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Enforce non-negativity on endmembers
                self.decoder.weight.data[self.decoder.weight <= 0] = 0
                self.decoder.weight.data[self.decoder.weight >= 1] = 1

            progress.set_postfix_str(f"loss={running_loss:.2e}")

        self.eval()
        with torch.no_grad():
            abund, _ = self(torch.Tensor(Y.T).to(self.device))
            Ahat = abund.cpu().numpy().T
            Ehat = self.decoder.weight.detach().cpu().numpy()

        self.time = time.time() - tic
        logger.info(f"ADMMNet took {self.time:.2f} seconds...")

        return Ehat, Ahat


class VCA:
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        L, N = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)

        #############################################
        # SNR Estimates
        #############################################

        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                :, :p
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            logger.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            logger.info(f"input SNR = {SNR}[dB]\n")

        SNR_th = 15 + 10 * np.log10(p)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            logger.info("... Select proj. to R-1")

            d = p - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            logger.info("... Select the projective proj.")

            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((p), dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1

        for i in range(p):
            w = generator.random(size=(p, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        logger.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        L, N = Y.shape  # L number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x**2) / float(N) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est
