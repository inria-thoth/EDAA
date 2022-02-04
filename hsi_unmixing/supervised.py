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
    print("Hello World!")
    logger.info(f"Current working directory: {os.getcwd()}")
    # print(OmegaConf.to_yaml(cfg))
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)

    results = {}

    for run in range(cfg.runs):
        hsi = instantiate(
            cfg.dataset,
            setter=setter,
            normalizer=normalizer,
        )
        # noise = AWGN()
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
        logging.info(f"aRMSE: {res:.2f}")
        hsi.plot_abundances(A0=A0)
        results[run] = res

    logging.info(results)
