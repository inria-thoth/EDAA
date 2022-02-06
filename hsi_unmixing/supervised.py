import json
import logging
import os
import pdb
import time

import numpy as np
from hydra.utils import instantiate

from hsi_unmixing.models.metrics import aRMSE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)

    results = []

    for run in range(cfg.runs):
        hsi = instantiate(
            cfg.dataset,
            setter=setter,
            normalizer=normalizer,
        )
        hsi.Y = noise.fit_transform(hsi.Y, SNR=cfg.SNR, seed=run)

        E0 = initializer.init_like(hsi, seed=run)

        aligner = instantiate(
            cfg.aligner,
            hsi=hsi,
            criterion=criterion,
        )
        E1 = aligner.fit_transform(E0)

        A0 = model.solve(hsi.Y, E1)

        metric = aRMSE()
        res = metric(hsi.A, A0)
        logger.info(f"aRMSE: {res:.2f}")

        if cfg.display:
            hsi.plot_abundances(A0=A0)

        results.append(res)

    logger.info(np.round(results, 2))
    mean = np.mean(results)
    std = np.std(results)
    logger.info(f"Mean +/- std [SNR={cfg.SNR}dB]: {mean:.2f} +/- {std:.2f} %")
