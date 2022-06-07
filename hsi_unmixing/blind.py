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
    # model = instantiate(cfg.model)
    noise = instantiate(cfg.noise)
    criterion = instantiate(cfg.criterion)

    RMSE = RMSEAggregator()
    SAD = SADAggregator()

    for run in range(cfg.runs):
        model = instantiate(cfg.model)
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

        E0, A0 = model.solve(
            Y,
            hsi.p,
            E0=E0,
            # E0=None,
            hsi=hsi,
            H=hsi.H,
            W=hsi.W,
            seed=run,
            aligner=aligner,
        )

        sparsity = (A0 <= 0.01).sum() / A0.size
        sparsity_printable = round(sparsity, 2)
        logger.info(f"Sparsity => {sparsity_printable}")

        E1 = aligner.fit_transform(E0)
        A1 = aligner.transform_abundances(A0)

        RMSE.add_run(run, hsi.A, A1, hsi.labels)
        SAD.add_run(run, hsi.E, E1, hsi.labels)

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
            X1 = aligner.transform_abundances(model.Xmap)
            hsi.plot_contributions(
                X0=X1,
                method=model,
                display=cfg.display,
                run=run,
            )

    RMSE.aggregate()
    SAD.aggregate()
