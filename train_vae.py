# import os
# import torch
from torch import optim
from models.vae import BaseVAE, VanillaVAE
import pytorch_lightning as pl
# from torchvision import transforms
# import torchvision.utils as vutils
from torch.utils.data import DataLoader
from typing import List, TypeVar, Any

Tensor = TypeVar('torch.tensor')


class VAETrainer(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAETrainer, self).__init__()

        self.model = VanillaVAE()
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
                *results,
                M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                optimizer_idx=optimizer_idx,
                batch_idx=batch_idx
                )

        self.log_dict(
                {f"val_{key}": val.item() for key, val in val_loss.items()},
                sync_dist=True
                )

    def on_validation_end(self) -> None:
        self.sample_images()

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(
                optims[0],
                gamma=self.params['scheduler_gamma'])
        scheds.append(scheduler)

        return optims, scheds
