import logging
import os
import pdb
import shutil

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="hsi_unmixing/config", config_name="config")
def main(cfg: DictConfig) -> None:

    if os.path.exists("config.yaml"):
        logger.info("Loading pre-existing config file")
        hydra_cfg = HydraConfig.get()
        overrides = hydra_cfg.overrides.task
        cfg = hydra.compose("config.yaml", overrides=overrides)
    else:
        # copy initial config to a separate file to avoid overwriting it
        # when hydra resumes training and initializes again
        shutil.copy2("./hydra/config.yaml", "config.yaml")

    mode = cfg.mode

    if mode == "blind":
        from hsi_unmixing.blind import main as _main
    elif mode == "supervised":
        from hsi_unmixing.supervised import main as _main
    else:
        raise ValueError(f"Mode {mode} is invalid")

    try:
        _main(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
