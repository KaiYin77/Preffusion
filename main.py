import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as transforms

import yaml
from pathlib import Path
import datetime

from dataset import Argoverse2Dataset, argo_multi_agent_collate_fn
from utils.get_opts import get_opts, create_yaml_parser
from utils.utils import generate_linear_schedule
from utils.visualize import VisualizeInterface
from utils.argo_eval import ArgoEval

from models.unet import UNet
# from models.ema import *
from models.ddpm import GaussianDiffusion
from models.vae import VanillaVAE
from train_vae import VAETrainer
# from copy import deepcopy
# import ipdb


class DDPMSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(DDPMSystem, self).__init__()
        self.save_hyperparameters(hparams)
        self.config = create_yaml_parser()

        self.model = UNet(
            img_channels=self.hparams.channel,
            base_channels=128,
            channel_mults=(1, 2),
            time_emb_dim=128*4,
            norm='gn',
            dropout=.1,
            activation=F.silu,
            attention_resolutions=(1,),
            num_classes=None,
            initial_pad=0,
            switch=self.hparams.switch,
        )
        self.vae_trainer = VAETrainer.load_from_checkpoint(
            self.hparams.vae_weight
        )
        self.vae_trainer.freeze()
        self.vae = self.vae_trainer.model

        betas = generate_linear_schedule(
            1000,
            1e-4 * 1000 / 1000,
            0.02 * 1000 / 1000,
        )
        self.ddpm = GaussianDiffusion(
            self.model, (60, 4), 1, 10,
            betas,
            ema_decay=.9999,
            ema_update_rate=1,
            ema_start=2000,
            loss_type='l2',
        )
        self.file_idx = 1
        self.visualize_interface = VisualizeInterface()
        self.argo_eval = ArgoEval()

    def forward(self, batch):
        ''' Future Trajectory
        '''
        x = batch['y'].reshape(-1, 60*5)
        # encode to latent space -> N, 256
        x = self.vae.encode(x)[0].reshape(-1, 1, 16, 16)
        ''' Conditioning Factor
        '''
        past_traj = batch['x'].reshape(-1, 300)
        lane = batch['lane_graph']
        neighbor = batch['neighbor_graph'].reshape(-1, 66)
        lane_mask = batch['lane_mask']
        neighbor_mask = batch['neighbor_mask']
        condition_fact = {'past_traj': past_traj,
                          'lane': lane,
                          'lane_mask': lane_mask,
                          'neighbor': neighbor,
                          'neighbor_mask': neighbor_mask}

        loss = self.ddpm(x, condition_fact)

        return loss

    def setup(self, stage):
        processed_train_dir = Path(self.config['data']['root']) /\
            Path('processed/training/')
        processed_train_dir.mkdir(parents=True, exist_ok=True)
        processed_val_dir = Path(self.config['data']['root']) /\
            Path('processed/validation/')
        processed_val_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset = Argoverse2Dataset(
            Path(self.config['data']['root'])/Path('raw/training/'),
            self.config['data']['training_txt'],
            processed_train_dir
        )
        self.val_dataset = Argoverse2Dataset(
            Path(self.config['data']['root'])/Path('raw/validation/'),
            self.config['data']['validation_txt'],
            processed_val_dir,
            mode="sampling"
        )
        self.collate_fn = argo_multi_agent_collate_fn

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.ddpm.parameters(),
            lr=self.hparams.lr,
            # weight_decay=self.hparams.weight_decay,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=len(self.train_dataloader()),
        #     eta_min=0,
        #     last_epoch=-1,
        # )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=10,
        #     gamma=0.9
        # )

        return self.optimizer
        # return [self.optimizer]
        # return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=0,
        )

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('lr', (self.optimizer).param_groups[0]['lr'])

        self.ddpm.update_ema()
        return loss

    def training_epoch_end(self, outputs):
        loss = [i['loss'].item() for i in outputs]
        loss = np.array(loss)
        self.log('train/loss', loss.mean())

    def validation_step(self, batch, batch_idx):
        samples = self.ddpm.sample(
            batch['noise_data'].shape[0],
            device=self.device,
            input=batch
        )
        samples = self.vae.decode(samples)
        samples = samples.reshape(10, -1, 1, 60, 5).squeeze(2).cpu().detach()
        gt = batch['y'].reshape(60, 5).cpu().detach()

        eval_result = self.argo_eval.forward(samples[-1], gt)
        self.log('val/ade', eval_result['min_ade'],
                 prog_bar=True, on_epoch=True)
        self.log('val/fde', eval_result['min_fde'],
                 prog_bar=True, on_epoch=True)
        return eval_result

    def validation_epoch_end(self, outputs):
        avg_min_ade = np.mean([output['min_ade'] for output in outputs])
        avg_min_fde = np.mean([output['min_fde'] for output in outputs])
        save_dir = Path('./evaluation/')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        f = open(f'{save_dir}/{fname}.txt', 'w')
        f.write('*** ARGO METRIC ***\n')
        f.write(f'total scenarios: {len(outputs)}\n')
        f.write(f'min ade: {avg_min_ade}\n')
        f.write(f'min fde: {avg_min_fde}\n')
        f.close()

    def test_step(self, batch, batch_idx):
        samples = self.ddpm.sample(
            batch['noise_data'].shape[0],
            device=self.device,
            input=batch
        )
        samples = self.vae.decode(samples)
        samples = samples.reshape(10, -1, 1, 60, 5).cpu().detach()
        self.visualize_interface.argo_forward(
            batch,
            batch_idx,
            self.test_dataset,
            samples,
        )
        return samples

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)

        if self.hparams.warmup_steps > 0:
            if (self.trainer.global_step <
                    self.hparams.warmup_steps *
                    self.trainer.num_training_batches):
                lr_scale = min(1.0, float(self.trainer.global_step + 1) /
                               float(self.hparams.warmup_steps
                                     * self.trainer.num_training_batches))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.lr


def main():
    hparams = get_opts()
    system = DDPMSystem(hparams)
    wandb_logger = WandbLogger(project='ddpm')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./ckpt/{hparams.exp_name}/',
        filename=hparams.exp_name+'/{epoch}',
        monitor='train/loss',
        mode='min',
        save_top_k=1,
    )
    if hparams.test:
        trainer = pl.Trainer(
            max_epochs=hparams.num_epoch,
            # default_root_dir='./ckpt/',
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            gpus=1,
            log_every_n_steps=1,
            fast_dev_run=hparams.fast_dev,
            gradient_clip_val=1,
        )
        model = system.load_from_checkpoint(hparams.weight)
        trainer.test(model)
    elif hparams.val:
        trainer = pl.Trainer(
            max_epochs=hparams.num_epoch,
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            gpus=1,
            log_every_n_steps=1,
            fast_dev_run=hparams.fast_dev,
            gradient_clip_val=1,
        )
        model = system.load_from_checkpoint(hparams.weight)
        trainer.validate(model)
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.num_epoch,
            # default_root_dir='./ckpt/',
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            gpus=1,
            log_every_n_steps=1,
            fast_dev_run=hparams.fast_dev,
            gradient_clip_val=1,
        )
        trainer.fit(system,
                    ckpt_path=hparams.weight)


if __name__ == '__main__':
    main()
