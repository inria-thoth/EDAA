import pdb
import os
import logging
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from hsi_unmixing import models
from hsi_unmixing import data

from hsi_unmixing.models import SparseCoding_pw

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="hsi_unmixing/config", config_name="config")
def train(cfg):

    # Load pre-existing config file
    if os.path.exists("config.yaml"):
        logging.info("Loading previous config file")
        cfg = OmegaConf.load("config.yaml")
    else:
        # copy config file into running directory
        shutil.copy2(".hydra/config.yaml", "config.yaml")

    # Check for checkpoint
    ckpt_path = os.path.join(os.getcwd(), cfg.checkpoint.dirpath, "last.ckpt")
    if os.path.exists(ckpt_path):
        logging.info("Loading existing checkpoint @ {ckpt_path}")
    else:
        logging.info("No existing ckpt found. Training from scratch")
        ckpt_path = None

    # Fix the seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Load data
    data_class = cfg.data.class_name
    dataset = data.__dict__[data_class](**cfg.data.params)

    # Use a Dataloader
    dataloader = DataLoader(dataset, cfg.data.batch_size, shuffle=True)

    # Load model
    # model_class = cfg.model.class_name
    # model = models.__dict__[model_class](**cfg.model.params)
    model = SparseCoding_pw(**cfg.model.params)

    # Use callbacks
    callbacks = [ModelCheckpoint(**cfg.checkpoint)]

    # Use Trainer
    trainer = pl.Trainer(resume_from_checkpoint=ckpt_path,
                         callbacks=callbacks,
                         **cfg.trainer.params)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical(e, exc_info=True)
