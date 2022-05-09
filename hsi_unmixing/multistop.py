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
        timesteps=[20, 50, None],
    )

    for key, stop in results.items():
        logger.debug(key, stop)

        RMSE = RMSEAggregator()
        SAD = SADAggregator()

        E0 = stop["E"]
        A0 = stop["A"]

        E1 = aligner.fit_transform(E0)
        A1 = aligner.transform_abundances(A0)

        RMSE.add_run(0, hsi.A, A1, hsi.labels)
        SAD.add_run(0, hsi.E, E1, hsi.labels)

        RMSE.aggregate(prefix=key)
        SAD.aggregate(prefix=key)

        # hsi.Y = noise.fit_transform(hsi.Y, SNR=cfg.SNR, seed=run)

        # E0 = initializer.init_like(hsi, seed=run)

        # if run == 0:
        #     hsi.plot_endmembers(display=cfg.display)
        #     hsi.plot_abundances(display=cfg.display)
        #     hsi.plot_PCA(display=cfg.display, E0=E0, initializer=True)

    # E1 = aligner.fit_transform(E0)
    # A1 = aligner.transform_abundances(A0)

    # RMSE.add_run(run, hsi.A, A1, hsi.labels)
    # SAD.add_run(run, hsi.E, E1, hsi.labels)

    # hsi.plot_endmembers(
    #     E0=E1,
    #     display=cfg.display,
    #     run=run,
    # )
    # hsi.plot_abundances(
    #     A0=A1,
    #     display=cfg.display,
    #     run=run,
    # )

    # hsi.plot_PCA(
    #     E0=E1,
    #     display=cfg.display,
    #     run=run,
    # )

    # if hasattr(model, "Xmap"):
    #     X1 = aligner.transform_abundances(model.Xmap)
    #     hsi.plot_contributions(
    #         X0=X1,
    #         method=model,
    #         display=cfg.display,
    #         run=run,
    #     )

    # RMSE.aggregate()
    # SAD.aggregate()
