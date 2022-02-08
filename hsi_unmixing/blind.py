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

    # Instantiate modules objects
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)
    # TODO Add SAD for endmembers
    metric = aRMSE()
    # metrics = [aRMSE(), SAD()]
    # TODO Add multiple metrics

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
        Y, _, _ = hsi(asTensor=cfg.torch)

        E0, A0 = model.solve(Y, hsi.p, E0=E0)

        # if cfg.torch:
        #     E0 = E0.detach().numpy()
        #     A0 = A0.detach().numpy()

        E1 = aligner.fit_transform(E0)
        A1 = aligner.transform_abundances(A0)

        res = metric(hsi.A, A1)
        logger.info(f"aRMSE: {res:.2f}")

        hsi.plot_endmembers(
            E0=E1,
            display=cfg.display,
            run=run,
        )
        hsi.plot_abundances(
            A0=A1,
            display=cfg.display,
            run=run,
        )

        if hasattr(model, "Xmap"):
            X1 = aligner.transform_abundances(model.Xmap)
            hsi.plot_contributions(
                X0=X1,
                method=model,
                display=cfg.display,
                run=run,
            )

        results.append(res)

    logger.info(np.round(results, 2))
    mean = np.mean(results)
    std = np.std(results)
    logger.info(f"Mean +/- std [SNR={cfg.SNR}dB]: {mean:.2f} +/- {std:.2f} %")
