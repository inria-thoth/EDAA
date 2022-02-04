import json
import logging
import os
import pdb
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf

# from hsi_unmixing.data import AWGN
# from hsi_unmixing.models.initializers import VCA
# from hsi_unmixing import data as data_utils
# from hsi_unmixing.data import datasets
from hsi_unmixing.models.metrics import aRMSE

# from hsi_unmixing.models.supervised import FCLS, DecompSimplex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    logger.info(f"Current working directory: {os.getcwd()}")
    # print(OmegaConf.to_yaml(cfg))
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)
    metric = aRMSE()

    results = {}

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
        E0, A0 = model.solve(hsi.Y, hsi.p, E0=E0)

        E1 = aligner.fit_transform(E0)
        A1 = aligner.transform_abundances(A0)

        res = metric(hsi.A, A1)
        logging.info(f"aRMSE: {res:.2f}")
        hsi.plot_endmembers(E0=E1)
        hsi.plot_abundances(A0=A1)

        if hasattr(model, "Xmap"):
            X1 = aligner.transform_abundances(model.Xmap)
            hsi.plot_contributions(X0=X1, method=model)

        results[run] = res

    logging.info(results)
