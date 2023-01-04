'''
Boosted by lightning module
'''
import os
import copy
import datetime
from pathlib import Path
import wandb

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import Argoverse2Dataset, argo_multi_agent_collate_fn
from utils.get_opts import get_opts, create_yaml_parser

args = get_opts()
config = create_yaml_parser()

root = Path(config['data']['root'])
val_txt = config['data']['validation_txt']

val_dir = Path(root) / Path('raw/validation')
processed_val_dir = Path(root) / Path('processed/validation/')
processed_val_dir.mkdir(parents=True, exist_ok=True)

ckpt_dir = Path('weights/')
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = Path(str(args.weight))

class STPvCDM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dataset = Argoverse2Dataset
        self.collate_fn = argo_multi_agent_collate_fn

        # [SHOULD BE REPLACE] MOCK MODEL
        self.model = nn.Linear(2, 128, 128)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr)
        return [self.optimizer]

    def training_step(self, batch, idx):
        ''' Future Trajectory
        '''
        x = batch['y'].reshape(-1, 300)
        
        ''' Conditioning Factor
        '''
        past_traj = batch['x'].reshape(-1, 300)
        lane = batch['lane_graph']
        neighbor = batch['neighbor_graph'].reshape(-1, 66)
        
        loss = torch.zeros(1, 1).requires_grad_()
        self.log('loss', loss.item(), prog_bar=True)
        return loss

    def prepare_data(self):
        self.train_dataset = self.dataset(val_dir, val_txt, processed_val_dir)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            collate_fn=self.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )

if __name__ == '__main__':
    wandb_logger = WandbLogger(
        project="STP-via-CDM")
    if args.train:
        model = STPvCDM()
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch:02d}-{loss:.2f}',
            save_top_k=args.save_top_k, 
            mode="min",
            monitor="loss"
        )
        trainer = pl.Trainer(
            callbacks=[
                checkpoint_callback],
            accelerator="gpu",
            max_epochs=args.num_epoch,
            logger=wandb_logger,
            gradient_clip_val=1,
            track_grad_norm=2,
            fast_dev_run=True if args.fast_dev else False
        )
        if args.weight:
            trainer.fit(
                model, ckpt_path=ckpt_path)
        else:
            trainer.fit(
                model)
