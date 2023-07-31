import logging

from hydra.utils import instantiate

<<<<<<< HEAD
from hsi_unmixing.models.metrics import (
    ARMSEAggregator,
    SADAggregator,
    ERMSEAggregator,
)
=======
from hsi_unmixing.models.metrics import RMSEAggregator, SADAggregator
>>>>>>> 676a96c56d8d4dab905e83b7996d748c3975f3e0
from hsi_unmixing.utils import save_estimates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cfg):

    # Instantiate modules objects
    setter = instantiate(cfg.setter)
    normalizer = instantiate(cfg.normalizer)
    initializer = instantiate(cfg.initializer)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)

<<<<<<< HEAD
    ARMSE = ARMSEAggregator()
    SAD = SADAggregator()
    ERMSE = ERMSEAggregator()
=======
    RMSE = RMSEAggregator()
    SAD = SADAggregator()
>>>>>>> 676a96c56d8d4dab905e83b7996d748c3975f3e0

    for run in range(cfg.runs):
        model = instantiate(cfg.model)
        hsi = instantiate(
            cfg.dataset,
            setter=setter,
            normalizer=normalizer,
        )
        hsi.Y = noise.fit_transform(hsi.Y, SNR=cfg.SNR, seed=run)

        if run == 0:
            hsi.plot_endmembers(display=cfg.display)
            hsi.plot_abundances(display=cfg.display)

        aligner = instantiate(
            cfg.aligner,
            hsi=hsi,
            criterion=criterion,
        )

        Y, _, _ = hsi(asTensor=cfg.torch)

        E0, A0 = model.solve(
            Y,
            hsi.p,
            seed=run,
            H=hsi.H,
            W=hsi.W,
        )

        A1 = aligner.fit_transform(A0)
        E1 = aligner.transform_endmembers(E0)

<<<<<<< HEAD
        ARMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)
        ERMSE.add_run(run, hsi.scaledE, E1, hsi.labels)
=======
        RMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)
>>>>>>> 676a96c56d8d4dab905e83b7996d748c3975f3e0

        if cfg.save_figs:

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

            if hasattr(model, "Xmap"):
                X1 = aligner.transform(model.Xmap)
                hsi.plot_contributions(
                    X0=X1,
                    method=model,
                    display=cfg.display,
                    run=run,
                )

<<<<<<< HEAD
    ARMSE.aggregate()
    SAD.aggregate()
    ERMSE.aggregate()
=======
    RMSE.aggregate()
    SAD.aggregate()
>>>>>>> 676a96c56d8d4dab905e83b7996d748c3975f3e0

    # Save last estimates
    save_estimates(E1, A1, hsi)
