import logging

from hydra.utils import instantiate, to_absolute_path

import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# from hsi_unmixing.models.metrics import (
#     ARMSEAggregator,
#     SADAggregator,
#     ERMSEAggregator,
# )
# from hsi_unmixing.utils import save_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):

    # TODO Single run
    model = instantiate(cfg.model)

    path = to_absolute_path("./data/Cuprite.mat")

    data = sio.loadmat(path)

    Y = data["Y"]
    H = int(data["H"])
    W = int(data["W"])
    p = cfg.p
    # TODO L2 Normalization??
    if cfg.normalize:
        Y = Y / np.linalg.norm(Y, axis=0, keepdims=True)
    if cfg.torch:
        Y = torch.Tensor(Y)
    E0, A0 = model.solve(Y, p, seed=cfg.seed)

    logger.debug(f"E0 shape: {E0.shape}")
    logger.debug(f"A0 shape: {A0.shape}")

    plt.figure(figsize=(16, 4))
    plt.plot(E0)
    plt.savefig("./endmembers.png")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=p, figsize=(16, 4))
    for ii in range(p):
        mappable = ax[ii].imshow(A0[ii].reshape(H, W), vmin=0.0, vmax=A0[ii].max())
        ax[ii].axis("off")
        fig.colorbar(
            mappable,
            ax=ax[ii],
            location="right",
            shrink=0.33,
        )

    # plt.savefig("./abundances.png")
    fig.savefig("./abundances.png")
    plt.clf()
    plt.close()
