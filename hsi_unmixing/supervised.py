import logging

import torch
from hydra.utils import instantiate

from hsi_unmixing.models.metrics import ARMSEAggregator, SADAggregator, ERMSEAggregator
from hsi_unmixing.utils import save_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)

    ARMSE = ARMSEAggregator()
    SAD = SADAggregator()
    ERMSE = ERMSEAggregator()

    for run in range(cfg.runs):
        hsi = instantiate(
            cfg.dataset,
            setter=setter,
            normalizer=normalizer,
        )
        hsi.Y = noise.fit_transform(hsi.Y, SNR=cfg.SNR, seed=run)

        E0 = initializer.init_like(hsi, seed=run)

        if run == 0:
            hsi.plot_endmembers(display=cfg.display)
            hsi.plot_abundances(display=cfg.display)
            hsi.plot_PCA(display=cfg.display, E0=E0, initializer=True)

        aligner = instantiate(
            cfg.aligner,
            hsi=hsi,
            criterion=criterion,
        )

        Y, _, _ = hsi(asTensor=cfg.torch)

        if cfg.torch:
            E0 = torch.Tensor(E0)

        A0 = model.solve(Y, E0)

        A1 = aligner.fit_transform(A0)
        E1 = aligner.transform_endmembers(E0)

        if cfg.torch:
            A1 = A1.detach().numpy()

        ARMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)
        ERMSE.add_run(run, hsi.scaledE, E1, hsi.labels)

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
        hsi.plot_PCA(
            E0=E1,
            display=cfg.display,
            run=run,
        )

    ARMSE.aggregate()
    SAD.aggregate()
    ERMSE.aggregate()

    # NOTE Save last estimates
    save_estimates(E1, A1, hsi)
