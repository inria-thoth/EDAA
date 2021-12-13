import logging
import os
import pdb
import shutil
import time

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from hsi_unmixing import data, models
from hsi_unmixing.models import SparseCoding_pw, SC_ASC_pw, EDA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_model(model, dataloader, optimizer, device="cpu"):

    model = model.to(device)

    model.train()

    training_loss = 0.0

    for batch in dataloader:
        pixel, abund = batch
        pixel = pixel.to(device)
        abund = abund.to(device)
        optimizer.zero_grad()
        pixel_hat, codes = model(pixel)
        rec_loss = F.mse_loss(pixel_hat, pixel)
        loss = rec_loss
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    print(" training loss: %5.2f" % (training_loss))

    return model


def evaluate_model(model, dataloader, device="cpu"):

    model = model.to(device)

    model.eval()
    validation_loss = 0.0

    for idx, batch in enumerate(dataloader):
        pixel, abund = batch
        pixel = pixel.to(device)
        abund = abund.to(device)
        pixel_hat, codes = model(pixel)
        loss = F.mse_loss(codes, abund)
        validation_loss += loss.item()
    print(" validation loss: %5.2f" % (validation_loss))


dset = data.SimulatedDataCubes()


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

    train_dataloader = DataLoader(dset, cfg.data.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dset, cfg.data.batch_size, shuffle=False)
    model = EDA(**cfg.model.params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 450
    for ee in range(epochs):
        trained_model = train_model(
            model,
            train_dataloader,
            optimizer,
            device,
        )
        evaluate_model(trained_model, valid_dataloader, device)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical(e, exc_info=True)
