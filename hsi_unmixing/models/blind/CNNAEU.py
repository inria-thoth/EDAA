import logging
import time

import numpy as np
import scipy.sparse as sp
import spams
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CNNAEU(nn.Module):
    def __init__(
        self,
        p,  # nb of endmembers
        L,  # nb of channels
        scale=3,  # softmax scaling
        # num_patches=250,  # max nb of patches to select
    ):
        super().__init__()

        self.scale = scale
        # self.num_patches = num_patches
        lrelu_params = {"inplace": True, "negative_slope": 0.02}

        self.encoder = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(
                L,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.LeakyReLU(**lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(
                48,
                p,
                kernel_size=1,
                bias=False,
            ),
            nn.LeakyReLU(**lrelu_params),
            nn.BatchNorm2d(p),
            nn.Dropout2d(p=0.2),
        )
        self.decoder = nn.Conv2d(
            p,
            L,
            # kernel_size=13,
            kernel_size=11,
            # padding=6,
            padding=5,
            padding_mode="reflect",
            bias=False,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # breakpoint()
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        return abund, x_hat

    @staticmethod
    def loss(original, reconstruction):
        assert original.shape == reconstruction.shape

        dot_product = (original * reconstruction).sum(dim=1)
        original_norm = original.norm(dim=1)
        reconstruction_norm = reconstruction.norm(dim=1)
        sad_score = torch.clamp(
            dot_product / (original_norm * reconstruction_norm), -1, 1
        ).acos()
        return sad_score.mean()

    # @staticmethod
    # def loss(original, reconstruction):
    #     return ((original - reconstruction) ** 2).mean()

    def solve(
        self,
        Y,
        p,
        E0,
        hsi,
        epochs=320,
        # epochs=50,
        lr=0.0003,
        # num_patches=250,
        batch_size=15,
        patch_size=40,
        seed=0,
        **kwargs,
    ):

        torch.manual_seed(seed)
        print("Training started...")
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

        # Extract 2d patches
        l, h, w = hsi.L, hsi.H, hsi.W
        Y_numpy = Y.view(l, h, w).detach().numpy()
        Y_numpy = Y_numpy.transpose((1, 2, 0))

        # Compute num_patches to be extracted based on Urban
        num_patches = int(250 * (l * h * w) / (307 * 307 * 162))

        logging.info(f"{num_patches} patches extracted...")

        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=num_patches,
            patch_size=(patch_size, patch_size),
        )

        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        dataloader = torch.utils.data.DataLoader(
            input_patches,
            batch_size=batch_size,
            shuffle=True,
        )

        self = self.to(self.device)

        for ee in tqdm(range(epochs)):

            running_loss = 0
            for ii, batch in enumerate(dataloader):

                batch = batch.to(self.device)

                optimizer.zero_grad()

                abunds, outputs = self(batch)

                loss = self.loss(batch, outputs)
                running_loss += loss.item()
                loss.backward()

                optimizer.step()

            logger.debug(round(running_loss, 4))

        self.eval()

        # Get final abundances
        Y_eval = Y.view(1, l, h, w)
        Y_eval = Y_eval.to(self.device)

        abund, out = self(Y_eval)

        out = out.detach().cpu().numpy().reshape(l, h * w)
        A = abund.detach().cpu().numpy().reshape(p, h * w)

        # Get final endmembers
        E = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

        # Use decompSimplex here
        # Yf = np.asfortranarray(out, dtype=np.float64)
        # Ef = np.asfortranarray(E, dtype=np.float64)

        # W = spams.decompSimplex(
        #     Yf,
        #     Ef,
        #     computeXtX=True,
        #     numThreads=-1,
        # )

        # A = sp.csr_matrix.toarray(W)
        # A = np.asarray(W.todense())

        return E, A


def check_loss():
    B, L, H, W = 15, 20, 10, 10
    p = 10
    original = torch.rand(B, L, H, W)
    reconstruction = torch.rand(B, L, H, W)
    net = CNNAEU(p, L)
    sad1 = net.loss(original, original)
    print(f"SAD1 => {sad1}")
    sad2 = net.loss(original, 2 * original)
    print(f"SAD2 => {sad2}")
    sad3 = net.loss(original, -1 * original)
    print(f"SAD3 => {sad3}")
    sad4 = net.loss(original, reconstruction)
    print(f"SAD4 => {sad4}")
    print("Loss test passed...")


def check_forward():
    B, L, H, W = 15, 20, 10, 10
    p = 3
    faker = torch.rand(B, L, H, W)
    net = CNNAEU(p, L)
    output = net(faker)
    print("Forward test passed...")


if __name__ == "__main__":
    check_loss()
    # check_forward()
