import logging
import pdb
import os

import hydra
from hydra.utils import to_absolute_path
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from hsi_unmixing import data
from hsi_unmixing import models
from hsi_unmixing.utils.viz import plot_endmembers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(config_path="hsi_unmixing/config", config_name="config")
def extract(cfg):
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device used: {device}")

    # Assess checkpoint existence
    if cfg.load_ckpt is None:
        logger.info("No checkpoint submitted")
        return

    # Fix seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Load data
    data_class = cfg.data.class_name
    dataset = data.__dict__[data_class](**cfg.data.params)
    img_data = dataset.train_data
    img = torch.Tensor(img_data.astype("float32"))

    # Search for checkpoint
    ckpt_path = to_absolute_path(cfg.load_ckpt)

    if os.path.exists(ckpt_path):
        logger.info(f"Loading existing ckpt @ {ckpt_path}")
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_path, "config.yaml"))
        logger.info(f"Checkpoint config: \n{OmegaConf.to_yaml(ckpt_cfg)}")
    else:
        raise FileNotFoundError("No existing ckpt found @ {ckpt_path}")

    # Load model checkpoint
    model_class = ckpt_cfg.model.class_name
    model = models.__dict__[model_class](**ckpt_cfg.model.params)

    # Set model to evaluation mode
    model.plot_abundances(img, save=False)
    pdb.set_trace()
    # model.plot_endmembers(save=True)
    plot_endmembers(model, dataset)
    pdb.set_trace()





if __name__ == "__main__":
    extract()
