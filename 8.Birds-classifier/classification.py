import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, random_split
import torchvision
from torchvision import datasets, transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from PIL import ImageFile
from PIL import Image
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_curve
import torchvision.models as models
from sklearn.preprocessing import label_binarize
import cv2
from typing import Tuple, List, Dict, Optional, Any
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split


BATCH_SIZE = 16
MAX_EPOCHS = 200
BASE_LR = 1e-4


class BirdsClassifierDataset(Dataset):
    def __init__(self, images_path, targets, method, transform=None):
        self.images_path = images_path
        self.method = method
        if method == 'train':
            self.image_files = np.sort(os.listdir(images_path))
        self.targets = [
            targets[image_file] for image_file in self.image_files
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file, target = self.image_files[index], self.targets[index]
        image = cv2.imread(os.path.join(self.images_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
            target = torch.tensor(target)
        return image, target


class BirdsClassifierModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_gt, images_path):
        super().__init__()
        self.batch_size = batch_size
        self.images_path = images_path
        self.train_targets = train_gt
        self.train_transform = A.Compose(
            [A.SmallestMaxSize(max_size=640),
             A.RandomResizedCrop(height=512, width=512),
             A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                        b_shift_limit=15, p=0.5),
             A.RandomBrightnessContrast(p=0.5),
             A.HorizontalFlip(p=0.5),
             A.CoarseDropout(),
             A.Normalize(),
             ToTensorV2()]
        )

    def train_dataloader(self):
        self.train_dataset = BirdsClassifierDataset(
            self.images_path, self.train_targets, 'train', self.train_transform)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, transfer=True, freeze='most'):
        super().__init__()

        self.resnet_model = models.resnet18(pretrained=False)

        linear_size = list(self.resnet_model.children())[-1].in_features

        self.resnet_model.fc = nn.Linear(linear_size, 256)
        self.relu = nn.LeakyReLU(inplace=True)
        self.last = nn.Linear(256, num_classes)

        for child in list(self.resnet_model.children()):
            for param in child.parameters():
                param.requires_grad = True

        if freeze == 'last':
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most':
            for child in list(self.resnet_model.children())[:-4]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze != 'full':
            raise NotImplementedError('Wrong freezing parameter')

    def forward(self, x):
        out = self.resnet_model(x)
        out = self.relu(out)
        out = self.last(out)
        return F.log_softmax(out, dim=1)


class LightningBirdsClassifier(pl.LightningModule):

    def __init__(self, lr_rate=BASE_LR, freeze='most'):
        super(LightningBirdsClassifier, self).__init__()

        self.model = ResNetClassifier(50, True, freeze)

        self.lr_rate = lr_rate

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, torch.squeeze(y))
        acc = torch.sum(logits.argmax(dim=1) == torch.squeeze(y)) / y.shape[0]

        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        pass

    def test_step(self, val_batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_rate)
        self.optimizer = optimizer
        steps_per_epoch = 192
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


def train_classifier(train_gt, train_images_dir, fast_train=False):
    img_folder = train_images_dir
    dm = BirdsClassifierModule(BATCH_SIZE, train_gt, img_folder)
    model = LightningBirdsClassifier()
    if fast_train:
        trainer = pl.Trainer(max_epochs=1, logger=False,
                             checkpoint_callback=False)
        trainer.fit(model, dm)
    else:
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                             logger=False, checkpoint_callback=False)
        trainer.fit(model, dm)

    return model


def classify(model_path, test_images_dir):
    model = LightningBirdsClassifier.load_from_checkpoint(model_path)
    transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2(),
        ],
    )
    img_classes = {}
    with torch.no_grad():
        for image_file in os.listdir(test_images_dir):
            image = cv2.imread(os.path.join(test_images_dir, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)["image"]
            prediction = model(image[None, ...])
            prediction = torch.argmax(prediction).item()
            img_classes[image_file] = prediction
    return img_classes
