import logging
import pdb

from hydra.utils import instantiate

from hsi_unmixing.models.metrics import RMSEAggregator, SADAggregator

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

    hsi = instantiate(
        cfg.dataset,
        setter=setter,
        normalizer=normalizer,
    )
    aligner = instantiate(
        cfg.aligner,
        hsi=hsi,
        criterion=criterion,
    )
    Y, _, _ = hsi(asTensor=cfg.torch)

    hsi.plot_endmembers(display=False)
    hsi.plot_abundances(display=False)

    results = model.solve(
        # E0, A0 = model.solve(
        Y,
        hsi.p,
        # E0=E0,
        # E0=None,
        hsi=hsi,
        # H=hsi.H,
        # W=hsi.W,
        seed=0,
        aligner=aligner,
        # timesteps=[10, 20, 30, 40],
        timesteps=[10, 20, 30],
        runner_on=cfg.wandb,
    )

    for key, stop in results.items():
        logger.debug(key, stop)

        RMSE = RMSEAggregator()
        SAD = SADAggregator()

        E0 = stop["E"]
        A0 = stop["A"]

        sparsity = (A0 <= 0.01).sum() / A0.size
        sparsity_printable = round(sparsity, 2)
        logger.info(f"{key} => {sparsity_printable} (sparsity)")

        E1 = aligner.fit_transform(E0)
        A1 = aligner.transform_abundances(A0)

        RMSE.add_run(0, hsi.A, A1, hsi.labels)
        SAD.add_run(0, hsi.E, E1, hsi.labels)

        RMSE.aggregate(prefix=key)
        SAD.aggregate(prefix=key)

        hsi.plot_endmembers(
            E0=E1,
            display=cfg.display,
            run=key,
        )
        hsi.plot_abundances(
            A0=A1,
            display=cfg.display,
            run=key,
        )
