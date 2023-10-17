# regular imports
import os
import re
import numpy as np

# pytorch related imports
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from PIL import ImageFile
from PIL import Image
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_curve
import torchvision.models as models
from sklearn.preprocessing import label_binarize
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional, Any

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Squeeze input by taking spatial mean.
        squeezed = torch.mean(x, dim=(2, 3))
        excitated = self.excitation(squeezed)
        return x * excitated[:, :, None, None]


class DetectorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=2),
            ConvBlock(out_channels, out_channels),
            SEBlock(out_channels),
        )
        self.skip_connection = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            activation=False,
        )

    def forward(self, x):
        return self.block(x) + self.skip_connection(x)


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Detector(pl.LightningModule):
    def __init__(self, learning_rate, criterion):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.resnet = models.resnet18(pretrained=False)
        self.backbone = torch.nn.Sequential(
            *(list(self.resnet.children())[:-3]))
        self.model = nn.Sequential(
            ConvBlock(256, 32),
            ConvBlock(32, 32),
            DetectorBlock(32, 64),
            DetectorBlock(64, 64),
            ConvBlock(64, 16, kernel_size=1, padding=0),
            Flatten(),
            nn.Linear(64, 14 * 2),
        )

    def forward(self, x):
        output = self.backbone(x)
        output = self.model(output).view(-1, 14, 2)
        return self._normalize(output, min_value=-0.5, max_value=1.5)

    def _normalize(self, x, min_value, max_value):
        assert max_value > min_value
        value_range = max_value - min_value
        return min_value + torch.sigmoid(x) * value_range

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.criterion(prediction, y)
        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.optimizer = optimizer
        steps_per_epoch = 188
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return [optimizer], [scheduler]


class DetectionDataset(Dataset):
    def __init__(self, images_path, targets, method, transform=None):
        self.images_path = images_path
        self.method = method
        if method == 'train':
            self.image_files = np.sort(os.listdir(images_path))

        self.targets = [
            targets[image_file].reshape(14, 2) for image_file in self.image_files
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file, target = self.image_files[index], self.targets[index]
        image = cv2.imread(os.path.join(self.images_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image, keypoints=target)
            image, target = transformed["image"], transformed["keypoints"]

            target = torch.tensor(target, dtype=torch.float32)
            image_height, image_width = image.shape[-2:]
            target[:, 0] /= image_height
            target[:, 1] /= image_width

        return image, target


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_gt, images_path):
        super().__init__()
        self.batch_size = batch_size
        self.images_path = images_path
        self.train_targets = train_gt
        self.train_transform = A.Compose(
            [
                A.PadIfNeeded(100, 100),
                A.RandomResizedCrop(100, 100, scale=(
                    0.9, 1.0), ratio=(0.9, 1.1)),
                A.Rotate(limit=30),
                A.Normalize(),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(
                format='xy', remove_invisible=False)
        )
        self.val_transform = A.Compose(
            [A.Resize(100, 100), A.Normalize(), ToTensorV2()],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False)
        )
        self.test_transform = A.Compose(
            [A.Resize(100, 100), A.Normalize(), ToTensorV2()],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False)
        )

    def train_dataloader(self):
        self.train_dataset = DetectionDataset(
            self.images_path, self.train_targets, 'train', self.train_transform)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)


def train_detector(targets, img_folder, fast_train=False):
    dm = FaceDataModule(32, targets, img_folder)
    model = Detector(1e-5, nn.MSELoss())
    if fast_train:
        trainer = pl.Trainer(max_epochs=1, logger=False,
                             checkpoint_callback=False)
        trainer.fit(model, dm)
    else:
        trainer = pl.Trainer(max_epochs=20, logger=False,
                             checkpoint_callback=False)
        trainer.fit(model, dm)
    return model


def detect(model_file, images_path):
    model = Detector.load_from_checkpoint(
        model_file, learning_rate=None, criterion=nn.MSELoss())
    image_files = os.listdir(images_path)
    transform = A.Compose(
        [A.Resize(100, 100), A.Normalize(), ToTensorV2()],

    )
    detected_points = {}
    with torch.no_grad():
        for image_file in image_files:
            image = cv2.imread(os.path.join(images_path, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            image = transform(image=image)["image"]
            prediction = model(image[None, ...])[0].numpy()
            prediction[:, 0] *= h
            prediction[:, 1] *= w
            detected_points[image_file] = prediction.flatten()

    return detected_points
