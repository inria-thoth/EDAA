import os
import pdb
import logging
import time
import shutil
import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from hsi_unmixing import models
from hsi_unmixing import data
from hsi_unmixing.models import SparseCoding_pw

from hsi_unmixing.models.losses import ASC_penalty

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def train_validate(model, dataloader, optimizer, epochs=300, device='cpu'):

    for epoch in range(1, epochs):

        model.train()
        training_loss = 0.0

        for batch in dataloader:
            pixel, abund = batch
            pixel = pixel.to(device)
            abund = abund.to(device)
            optimizer.zero_grad()
            pixel_hat, codes = model(pixel)
            loss = (torch.sum(F.mse_loss(pixel_hat, pixel)).float())# + (ASC_penalty(codes, 0.03).float()) 
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss  = training_loss / len(batch[0]) 

        model.eval()
        validation_loss = 0.0

        for batch in dataloader:
            pixel, abund = batch
            pixel = pixel.to(device)
            abund = abund.to(device)
            pixel_hat, codes = model(pixel)
            loss = (torch.sum(F.mse_loss(pixel_hat, pixel)).float())# + (ASC_penalty(codes, 0.03).float()) 
            validation_loss += loss.item()
        validation_loss  = validation_loss / len(batch[0]) 

        if epoch == 1 or epoch % 10 == 0:
          print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
                (epoch, epochs, training_loss, validation_loss))


dset = data.SimulatedPatches("./data")
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

   
    dataloader = DataLoader(dset, cfg.data.batch_size, shuffle=True)
    model = SparseCoding_pw(**cfg.model.params)
    optimizer =  torch.optim.Adam(model.parameters(), lr=0.001) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 250
    train_validate(model, 
                    dataloader, 
                    optimizer, 
                    epochs,
                    device)
    

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical(e, exc_info=True)
