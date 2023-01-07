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

# from pytorch_gan_metrics import get_fid
import yaml
from pathlib import Path

from dataset import Argoverse2Dataset, argo_multi_agent_collate_fn
from utils.get_opts import get_opts, create_yaml_parser
from utils.utils import generate_linear_schedule
from utils.visualize import VisualizeInterface

from models.unet import UNet
# from models.ema import *
from models.ddpm import GaussianDiffusion
from models.vae import VanillaVAE
from train_vae import VAETrainer
# from copy import deepcopy
import ipdb


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
                './ckpt/vae/vae_epoch=14.ckpt'
        )
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
        processed_val_dir = Path(self.config['data']['root']) /\
            Path('processed/validation/')
        processed_val_dir.mkdir(parents=True, exist_ok=True)
        self.train_dataset = Argoverse2Dataset(
            Path(self.config['data']['root'])/Path('raw/validation/'),
            self.config['data']['validation_txt'],
            processed_val_dir
        )
        self.test_dataset = Argoverse2Dataset(
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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=0,
        )

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # self.log("train/loss", loss)
        self.log('lr', (self.optimizer).param_groups[0]['lr'])

        self.ddpm.update_ema()

        return loss

    def training_epoch_end(self, outputs):
        loss = [i['loss'].item() for i in outputs]
        loss = np.array(loss)
        self.log('train/loss', loss.mean())

        # samples = self.ddpm.sample(8, self.device)
        # samples = self.ddpm.sample_diffusion_sequence(
        #         8,
        #         device=self.device)
        # samples = torch.cat(samples)
        # samples = (samples.clip(-1, 1)+1)/2
        # # ipdb.set_trace()
        # # samples = samples.view(-1, 3, 32, 32)
        # samples = transforms.Resize((28, 28))(samples)
        # save_image(samples, 'gen.png')
        #
        # samples = self.ddpm.sample(10, self.device)
        # samples = (samples.clamp(-1, 1) + 1) / 2
        # samples = transforms.Resize((28, 28))(samples)
        # FID = get_fid(samples, 'data/mnist.npz')
        # self.log('fid', FID)

    def test_step(self, batch, batch_idx):
        samples = self.ddpm.sample(
            batch['noise_data'].shape[0],
            device=self.device,
            input=batch
        )
        self.visualize_interface.argo_forward(
            batch,
            batch_idx,
            self.test_dataset,
            samples,
        )
        return samples

    # def test_epoch_end(self, outputs):
        # ipdb.set_trace()
        # all_samples = torch.cat(outputs)
        # all_samples = transforms.Resize((28, 28))(all_samples)

        # for i, img in tqdm(enumerate(all_samples)):
        #     save_image(img, f'generated/{(i+1):05d}.png')
        # FID = get_fid(all_samples, './data/mnist.npz')
        # print(f'FID: \t{FID}')

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
            # resume_from_checkpoint=hparams.weight,
        )
        # trainer.test(ckpt_path=hparams.weight)
        # # ipdb.set_trace()
        model = system.load_from_checkpoint(hparams.weight)
        trainer.test(model)
        # model.eval()

        # seq = model.ddpm.sample_diffusion_sequence(8, device=model.device)
        # seq = torch.cat(seq)
        # seq = (seq.clip(-1, 1) + 1) / 2
        # seq = transforms.Resize((28, 28))(seq)
        # # ipdb.set_trace()
        # seq[:8] = torch.zeros_like(seq[:8]) + 0.5
        # save_image(seq, 'process.png')

        # all_samples = []
        # for i in range(10):
        #     samples = model.ddpm.sample(1000, model.device)
        #     samples = (samples.clamp(-1, 1) + 1) / 2
        #     all_samples.append(samples)
        # all_samples = torch.cat(all_samples)

        # all_samples = transforms.Resize((28, 28))(all_samples)

        # for i, img in tqdm(enumerate(all_samples)):
        #     save_image(img, f'generated/{(i+1):05d}.png')
        # FID = get_fid(all_samples, './data/mnist.npz')
        # print(f'FID: \t{FID}')

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
