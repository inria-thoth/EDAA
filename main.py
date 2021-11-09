import os
import pdb
import logging
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

def train_validate(model, train_dataloader, valid_dataloader, optimizer, loss_function, epochs=300, device='cpu' ):

    for epoch in range(1, epochs):

        model.train()
        training_loss = 0.0

        for batch in train_dataloader:
            pixel, abund = train_dataloader
            pixel = pixel.to(device)
            abund = abund.to(device)
            optimizer.zero_grad()
            # propagate throught the model
                # pixel_hat, codes = model()...........
            # compute the losses
                # loss = loss_function()........
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        train_loss  = train_loss / len(train_dataloader.dataset) # or index to get the real lenght
        # Computer the means SAD too
        # train_SAD = 

        model.eval()
        validation_loss = 0.0

        for batch in valid_dataloader:
            pixel, abund = valid_dataloader
            pixel = pixel.to(device)
            abund = abund.to(device)
            # propagate throught the model
                # pixel_hat, codes = model()...........
            # compute the losses
                # loss = loss_function()........
            validation_loss += loss.item()
        validation_loss  = validation_loss / len(valid_dataloader.dataset) # or index to get the real lenght
        # Computer the means SAD too
        # val_SAD = 

        if epoch == 1 or epoch % 10 == 0:
          print('Epoch %3d/%3d, train loss: %5.2f, train SAD: %5.2f, val loss: %5.2f, val SAD: %5.2f' % \
                (epoch, epochs, train_loss, train_SAD, validation_loss, val_SAD))
# python -m hsi_unmixing.data.Simulated