import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models.unet.model import UNet
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import random

DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)


# Sweep parameters
hyperparameter_defaults = dict(
    data_path='data_semantics',
    batch_size = 2,
    lr = 1e-3,
    num_layers = 5,
    features_start = 64,
    bilinear = False,
    grad_batches = 1,
    epochs = 20
)

wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config



class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(self, hparams):
        super().__init__()
        self.lr = hparams.lr
        self.net = UNet(num_classes=19, num_layers=hparams.num_layers,
                        features_start=hparams.features_start, bilinear=hparams.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log('train_loss', loss_val)    # log training loss
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log('val_loss', loss_val)  # will be automatically averaged over an epoch
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


class KittiDataModule(pl.LightningDataModule):
    '''
    Kitti Data Module
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    '''

    def __init__(self, hparams):
        super().__init__()
        print(hparams)
        self.data_path = hparams.data_path
        self.batch_size = hparams.batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])
    
    def setup(self, stage=None):
        self.trainset = KITTI(self.data_path, split='train', transform=self.transform)
        self.validset = KITTI(self.data_path, split='valid', transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)


def main(config):
    # ------------------------
    # 1 LIGHTNING MODEL
    # ------------------------
    model = SegModel(config)

    # ------------------------
    # 2 DATA PIPELINES
    # ------------------------
    kittiData = KittiDataModule(config)

    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------
    wandb_logger = WandbLogger()

    # optional: log model topology
    wandb_logger.watch(model.net)

    # ------------------------
    # 4 TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=-1,
        logger=wandb_logger,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.grad_batches,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model, kittiData)


if __name__ == '__main__':

    print(f'Starting a run with {config}')
    main(config)