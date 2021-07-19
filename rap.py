# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from glob import glob
from os.path import join
import numpy as np
from pytorch_lightning import callbacks
import skimage.io as io
import torch
from skimage.transform import resize
from torchvision import transforms

from datasets.transforms import ToCrops
from datasets.transforms import ToFloatTensor3D


# %%
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, dataloader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
# from pl_bolts.datasets import DummyDataset

# %% [markdown]
# ---
# ## Data


# %%
from PIL import Image
import torch.utils.data as data
import os
import torchvision.transforms as transforms
import torch

class UCSDAnomalyDataset(data.Dataset):
    '''
    Dataset class to load  UCSD Anomaly Detection dataset
    Input: 
    - root_dir -- directory (Train/Test) structured exactly as out-of-the-box folder downloaded from the site
    http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
    - time_stride (default 1) -- max possible time stride used for data augmentation
    - seq_len -- length of the frame sequence
    Output:
    - tensor of 10 normlized grayscale frames stiched together
    
    Note:
    [mean, std] for grayscale pixels is [0.3750352255196134, 0.20129592430286292]
    '''
    def __init__(self, root_dir, seq_len = 16, time_stride=1, transform=None):
        super(UCSDAnomalyDataset, self).__init__()
        self.root_dir = root_dir
        vids = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.samples = []
        for d in vids:
            img_list = sorted(glob(join(root_dir + '/'+  d, '*.tif')))

            n_imgs = len(img_list)

            for t in range(1, time_stride+1):
                for i in range(1, n_imgs):
                    if i+(seq_len-1)*t > n_imgs:
                        break
                    self.samples.append((os.path.join(self.root_dir, d), range(i, i+(seq_len-1)*t+1, t)))
        self.pil_transform = transforms.Compose([
                    ToFloatTensor3D(),
                     ToCrops((1, 16, 256, 384), (1, 8, 32, 32))])

        
    def __getitem__(self, index):
        sample = []
        pref = self.samples[index][0]
        for fr in self.samples[index][1]:
            with open(os.path.join(pref, '{0:03d}.tif'.format(fr)), 'rb') as fin:
                img = io.imread(fin)
                img = resize(img, output_shape=(256, 384), preserve_range=True)
                frame = np.uint8(img)
                sample.append(frame)
        sample = np.stack(sample, axis=0)
        clip = np.expand_dims(sample, axis=-1)
        sample = clip, clip

        if self.pil_transform:
            sample = self.pil_transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

# %% [markdown]
# ---
# 
# ## Model

# %%
from models import LSAUCSD
# from models.loss_functions import LSALoss
from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss




import numpy as np
from utils import normalize
from utils import novelty_score
from glob import glob
import skimage.io as io
from os.path import join
from sklearn.metrics import roc_auc_score

# %%
class ResultsAccumulator:
    """
    Accumulates results in a buffer for a sliding window
    results computation. Employed to get frame-level scores
    from clip-level scores.
    ` In order to recover the anomaly score of each
    frame, we compute the mean score of all clips in which it
    appears`
    """
    def __init__(self, time_steps):
        # type: (int) -> None
        """
        Class constructor.

        :param time_steps: the number of frames each clip holds.
        """

        # This buffers rotate.
        self._buffer = np.zeros(shape=(time_steps,), dtype=np.float32)
        self._counts = np.zeros(shape=(time_steps,))

    def push(self, score):
        # type: (float) -> None
        """
        Pushes the score of a clip into the buffer.
        :param score: the score of a clip
        """

        # Update buffer and counts
        self._buffer += score
        self._counts += 1

    def get_next(self):
        # type: () -> float
        """
        Gets the next frame (the first in the buffer) score,
        computed as the mean of the clips in which it appeared,
        and rolls the buffers.

        :return: the averaged score of the frame exiting the buffer.
        """

        # Return first in buffer
        ret = self._buffer[0] / self._counts[0]

        # Roll time backwards
        self._buffer = np.roll(self._buffer, shift=-1)
        self._counts = np.roll(self._counts, shift=-1)

        # Zero out final frame (next to be filled)
        self._buffer[-1] = 0
        self._counts[-1] = 0

        return ret

    @property
    def results_left(self):
        # type: () -> np.int32
        """
        Returns the number of frames still in the buffer.
        """
        return np.sum(self._counts != 0).astype(np.int32)
# %%
import matplotlib.pyplot as plt
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = LSAUCSD(input_shape=(1, 8, 32, 32), code_length=64, cpd_channels=100)
        # self.loss_fn = LSALoss(cpd_channels=100)

        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss(100)

        self.results_accumulator_llk = ResultsAccumulator(time_steps=16)
        self.results_accumulator_rec = ResultsAccumulator(time_steps=16)

        self.sample_llk = np.zeros(shape=(150,))
        self.sample_rec = np.zeros(shape=(150,))





    def training_step(self, batch, batch_idx):
        x, y = batch

        x_r, z, z_dist = self.model(x.squeeze(0))


        rec_loss = self.reconstruction_loss_fn(x, x_r)
        arg_loss = self.autoregression_loss_fn(z, z_dist)
        tot_loss = rec_loss + 1 * arg_loss


        self.log('train_reconstruction_loss_fn', rec_loss)
        self.log('train_autoregression_loss', arg_loss)
        return tot_loss


    # def validation_step(self, batch, batch_idx):
    #     x, y = batch

    #     x_r, z, z_dist = self.model(x.squeeze(0))


    #     rec_loss = self.reconstruction_loss_fn(x, x_r)
    #     arg_loss = self.autoregression_loss_fn(z, z_dist)
    #     tot_loss = rec_loss + 1 * arg_loss


    #     self.log('val_reconstruction_loss_fn', rec_loss)
    #     self.log('val_autoregression_loss', arg_loss)
    #     return tot_loss


    def test_step(self, batch, batch_idx):
        x, _ = batch
        # import pdb;pdb.set_trace()
        # tensorboard = self.logger.experiment
        # tensorboard.add_image('org',x)

        x_r, z, z_dist = self.model(x.squeeze(0))

        # tensorboard.add_image('recon',x_r)

        rec_loss = self.reconstruction_loss_fn(x, x_r)
        arg_loss = self.autoregression_loss_fn(z, z_dist)
        tot_loss = rec_loss + 1 * arg_loss

        i = batch_idx
        self.results_accumulator_llk.push(arg_loss.item())
        self.results_accumulator_rec.push(rec_loss.item())
        self.sample_llk[i] = self.results_accumulator_llk.get_next()
        self.sample_rec[i] = self.results_accumulator_rec.get_next()


        self.log('test_reconstruction_loss', rec_loss.item(),on_step=True,on_epoch=False)
        self.log('test_autoregression_loss', arg_loss.item(),on_step=True,on_epoch=False)
        return tot_loss.item()
    
    def on_test_epoch_end(self):
        print('计算nvelty score')
        results_accumulator_llk = self.results_accumulator_llk
        results_accumulator_rec = self.results_accumulator_rec
        sample_llk = self.sample_llk
        sample_rec = self.sample_rec

        while results_accumulator_llk.results_left != 0:
            index = (- results_accumulator_llk.results_left)
            sample_llk[index] = results_accumulator_llk.get_next()
            sample_rec[index] = results_accumulator_rec.get_next()
        min_llk, max_llk= sample_llk.min(),sample_rec.min()
    
        min_rec, max_rec = sample_llk.max(),sample_rec.max()

        sample_llk = normalize(sample_llk, min_llk, max_llk)
        sample_rec = normalize(sample_rec, min_rec, max_rec)
        sample_ns = novelty_score(sample_llk, sample_rec)
        print(sample_ns)
        plt.plot(sample_ns)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# %%
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


tb_logger = TensorBoardLogger('logs/')
# init model
ae = LitAutoEncoder()
# wandb_logger = WandbLogger() 
# Initialize a trainer

trainer = pl.Trainer(gpus=1, max_epochs=10, progress_bar_refresh_rate=20,logger=tb_logger,
resume_from_checkpoint='/mnt/luowh/june/novelty-detection/wandb/run-20210714_173838-3bdchcum/files/novelty-detection/3bdchcum/checkpoints/epoch=9-step=23099.ckpt'
)

train = DataLoader(UCSDAnomalyDataset('data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'),num_workers =16,pin_memory=True)
test = DataLoader(UCSDAnomalyDataset('data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test1'),num_workers =16,pin_memory=True)
# Train the model ⚡
trainer.fit(ae, train)

trainer.test(ae,test_dataloaders=test)

# %%
