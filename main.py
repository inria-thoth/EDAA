import logging
import os
import pdb

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="hsi_unmixing/config", config_name="config")
def main(cfg: DictConfig) -> None:

    from hsi_unmixing.hello_world import main as _main

    _main(cfg)


if __name__ == "__main__":
    main()
