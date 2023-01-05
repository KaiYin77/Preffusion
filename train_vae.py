# import os
# import torch
from torch import optim
from models.vae import BaseVAE, VanillaVAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader
from typing import List, TypeVar, Any

from dataset import Argoverse2Dataset, argo_multi_agent_collate_fn
from utils.get_opts import create_yaml_parser
from pathlib import Path
import ipdb

Tensor = TypeVar('torch.tensor')


class VAETrainer(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAETrainer, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        # try:
        #     self.hold_graph = self.params['retain_first_backpass']
        # except:
        #     pass

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x = batch['y'].reshape(-1, 60*5)
        self.curr_device = x.device

        results = self.forward(x)
        train_loss = self.model.loss_function(
                *results,
                M_N=self.params['kld_weight'],
                optimizer_idx=optimizer_idx,
                batch_idx=batch_idx
                )

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)
        self.log('train/loss', train_loss['loss'], on_epoch=True)
        self.log('train/recon_loss', train_loss['Reconstruction_Loss'])
        self.log('train/kld_loss', train_loss['KLD'])

        return train_loss['loss']

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


def main():
    config = create_yaml_parser()
    processed_val_dir = Path(config['data']['root']) /\
        Path('processed/validation/')
    processed_val_dir.mkdir(parents=True, exist_ok=True)

    collate_fn = argo_multi_agent_collate_fn
    train_dataset = Argoverse2Dataset(
            Path(config['data']['root'])/Path('raw/validation/'),
            config['data']['validation_txt'],
            processed_val_dir
            )
    train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=4
            )

    model = VanillaVAE(
            in_channels=config['vae']['in_channels'],
            latent_dim=config['vae']['latent_dim'],
            )
    vae_trainer = VAETrainer(
            model,
            config['vae']
            )
    wandb_logger = WandbLogger(project='trajectory-vae')

    checkpoint_callback = ModelCheckpoint(
            dirpath='./ckpt/vae/',
            filename='vae_{epoch}',
            monitor='train/loss',
            mode='min',
            save_top_k=1,
            )

    trainer = pl.Trainer(
            gpus=1,
            max_epochs=config['vae']['epochs'],
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            )
    trainer.fit(vae_trainer, train_loader)


if __name__ == '__main__':
    main()
