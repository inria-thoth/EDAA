import logging
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MiSiCNet(nn.Module):
    def __init__(
        self,
        p,  # nb of endmembers
        L,  # nb of channels
    ):
        super().__init__()

        kernel_size = 3
        padding = (kernel_size - 1) // 2

        lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                L,
                256,
                kernel_size,
                # padding=padding,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**lrelu_params),
        )

        # self.layer12 = nn.Sequential(
        #     nn.ReflectionPad2d(padding),
        #     nn.Conv2d(
        #         256,
        #         256,
        #         kernel_size,
        #     ),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(**lrelu_params),
        # )

        self.skiplayer = nn.Sequential(
            # nn.ReflectionPad2d(0),
            nn.Conv2d(
                L,
                4,
                # 16,
                kernel_size=1,
            ),
            nn.BatchNorm2d(4),
            # nn.BatchNorm2d(16),
        )

        self.layer2 = nn.Sequential(
            # nn.Upsample(scale_factor=1),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                260,
                # 272,
                256,
                # p,
                kernel_size,
                # padding=padding,
            ),
            nn.BatchNorm2d(256),
            # nn.BatchNorm2d(p),
            nn.LeakyReLU(**lrelu_params),
            # nn.Softmax(dim=1),
        )

        self.layer3 = nn.Sequential(
            # nn.Upsample(scale_factor=1),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                256,
                p,
                kernel_size,
                # padding=padding,
            ),
            nn.BatchNorm2d(p),
            # nn.Softmax(dim=1),  # NOTE check this later
            # nn.Softmax(dim=0),
        )

        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.Linear(
            p,
            L,
            bias=False,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # @staticmethod
    # def plot(a):
    #     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    #     for kk in range(3):
    #         ax[kk].imshow(
    #             a[kk, :, :],
    #             vmin=0.0,
    #             vmax=1.0,
    #         )

    #     plt.show()

    def forward(self, x):
        _, c, h, w = x.shape
        # breakpoint()
        x1 = self.layer1(x)
        # x12 = self.layer12(x1)
        xskip = self.skiplayer(x)
        xcat = torch.cat([x1, xskip], 1)
        # xcat = torch.cat([x12, xskip], 1)
        x2 = self.layer2(xcat)
        x3 = self.layer3(x2)
        # x3 = x2
        # x4 = x3.view(x3.shape[1], -1).T
        x4 = self.softmax(x3)
        # x5 = x4.squeeze().reshape(h, w, -1).view(h * w, -1)
        x5 = torch.transpose(x4.squeeze().view(-1, h * w), 0, 1)
        # x5 = self.softmax(x4)
        out = self.decoder(x5)
        # out = F.linear(x5, self.decoder.weight.data)
        # return x3, out
        return x5, out

    def solve(
        self,
        Y,
        p,
        E0,
        hsi,
        # niters=8000,
        niters=8000,
        lr=0.001,
        exp_weight=0.99,
        # exp_weight=0,
        # lambd=100,
        lambd=100,
        **kwargs,
    ):
        # breakpoint()
        print("Solving started...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Reshape
        l, h, w = hsi.L, hsi.H, hsi.W
        Y = Y.view(1, l, h, w)

        # Use endmembers from VCA
        E_VCA = torch.Tensor(E0)
        self.decoder.weight.data = E_VCA

        self = self.to(self.device)
        Y = Y.to(self.device)
        noisy_input = torch.rand_like(Y)

        for ii in tqdm(range(niters)):
            optimizer.zero_grad()

            abund, reconst = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * exp_weight + abund.detach() * (1 - exp_weight)

            loss = self.loss(Y, self.decoder.weight, lambd, reconst)
            # loss = self.loss(Y, self.decoder.weight.data, lambd, reconst)

            if ii % 10 == 0:
                logger.debug(loss.item())

            loss.backward()
            optimizer.step()
            # Enforce physical constraints on endmembers
            self.decoder.weight.data[self.decoder.weight <= 0] = 0
            self.decoder.weight.data[self.decoder.weight >= 1] = 1

        # breakpoint()
        # Get final endmembers and abundances
        E = self.decoder.weight.detach().cpu().numpy()
        A = out_avg.cpu().T.numpy()

        return E, A

    @staticmethod
    def loss(target, endmembers, lambd, output):
        # breakpoint()
        N, L = output.shape
        _, l, h, w = target.shape
        target_reshape = target.squeeze().reshape(L, N)
        fit_term = 0.5 * torch.linalg.norm(target_reshape.t() - output, "fro") ** 2
        O = target_reshape.mean(1, keepdims=True)
        reg_term = torch.linalg.norm(endmembers - O, "fro") ** 2
        return fit_term + lambd * reg_term


def check_MiSiCNet_forward():
    H, W, L, p = 100, 100, 200, 5
    faker = torch.rand(1, L, H, W)
    net = MiSiCNet(p=p, L=L)
    out = net(faker)
    print("Check MiSiCNet forward passed...")


def check_MiSiCNet_solver():
    H, W, L, p = 100, 100, 200, 5
    faker = torch.rand(1, L, H, W)
    net = MiSiCNet(p=p, L=L)
    net.solve(faker, niters=5)
    # out = net(faker)
    print("Check MiSiCNet solver passed...")


def check_loss():
    print("Check Loss passed...")
    H, W, L, p = 100, 100, 200, 5
    lambd = 100
    target = torch.rand(1, L, H, W)
    _, l, h, w = target.shape
    out = torch.rand(H * W, L)
    E = torch.rand(L, p)

    # breakpoint()
    fit_term = 0.5 * torch.linalg.norm(target.view_as(out) - out, "fro") ** 2
    O = target.view(target.shape[1], -1).mean(1, keepdims=True)
    reg_term = torch.linalg.norm(E - O, "fro") ** 2

    return fit_term + lambd * reg_term


if __name__ == "__main__":
    check_MiSiCNet_forward()
    # check_loss()
    check_MiSiCNet_solver()
