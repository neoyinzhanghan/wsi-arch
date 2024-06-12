import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from wsiarch.models.components.hyena import HyenaOperator2D
from wsiarch.data.dataloaders import (
    FeatureImageDataset,
    create_data_loaders,
    FeatureImageDataModule,
)
from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import TensorBoardLogger

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import albumentations as A
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms, datasets, models
from torchmetrics import Accuracy, AUROC
from torch.utils.data import WeightedRandomSampler


############################################################################
####### DEFINE HYPERPARAMETERS AND DATA DIRECTORIES ########################
############################################################################

num_epochs = 500
default_config = {"lr": 3.56e-06}  # 1.462801279401232e-06}
data_dir = "/media/hdd1/neo/pooled_deepheme_data"
num_gpus = 3
num_workers = 20
downsample_factor = 1
batch_size = 256
img_size = 96
num_classes = 23

############################################################################
####### FUNCTIONS FOR DATA AUGMENTATION AND DATA LOADING ###################
############################################################################


def get_feat_extract_augmentation_pipeline(image_size):
    """Returns a randomly chosen augmentation pipeline for SSL."""

    ## Simple augumentation to improve the data generalizability
    transform_shape = A.Compose(
        [
            A.ShiftScaleRotate(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(shear=(-10, 10), p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.01),
                always_apply=False,
                p=0.2,
            ),
        ]
    )
    transform_color = A.Compose(
        [
            A.RandomBrightnessContrast(contrast_limit=0.4, brightness_limit=0.4, p=0.5),
            A.CLAHE(p=0.3),
            A.ColorJitter(p=0.2),
            A.RandomGamma(p=0.2),
        ]
    )

    # Compose the two augmentation pipelines
    return A.Compose(
        [A.Resize(image_size, image_size), A.OneOf([transform_shape, transform_color])]
    )


# Define a custom dataset that applies downsampling
class DownsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, downsample_factor, apply_augmentation=True):
        self.dataset = dataset
        self.downsample_factor = downsample_factor
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.downsample_factor > 1:
            size = (96 // self.downsample_factor, 96 // self.downsample_factor)
            image = transforms.functional.resize(image, size)

        # Convert image to RGB if not already
        image = to_pil_image(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.apply_augmentation:
            # Apply augmentation
            image = get_feat_extract_augmentation_pipeline(
                image_size=96 // self.downsample_factor
            )(image=np.array(image))["image"]

        image = to_tensor(image)

        return image, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, downsample_factor=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Additional normalization can be uncommented and adjusted if needed
                # transforms.Normalize(mean=(0.61070228, 0.54225375, 0.65411311), std=(0.1485182, 0.1786308, 0.12817113))
            ]
        )

    def setup(self, stage=None):
        # Load train, validation and test datasets
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "test"), transform=self.transform
        )

        # Prepare the train dataset with downsampling and augmentation
        self.train_dataset = DownsampledDataset(
            train_dataset, self.downsample_factor, apply_augmentation=True
        )
        self.val_dataset = DownsampledDataset(
            val_dataset, self.downsample_factor, apply_augmentation=False
        )
        self.test_dataset = DownsampledDataset(
            test_dataset, self.downsample_factor, apply_augmentation=False
        )

        # Compute class weights for handling imbalance
        class_counts_train = torch.tensor(
            [t[1] for t in train_dataset.samples]
        ).bincount()
        class_weights_train = 1.0 / class_counts_train.float()
        sample_weights_train = class_weights_train[
            [t[1] for t in train_dataset.samples]
        ]

        class_counts_val = torch.tensor([t[1] for t in val_dataset.samples]).bincount()
        class_weights_val = 1.0 / class_counts_val.float()
        sample_weights_val = class_weights_val[[t[1] for t in val_dataset.samples]]

        class_counts_test = torch.tensor(
            [t[1] for t in test_dataset.samples]
        ).bincount()
        class_weights_test = 1.0 / class_counts_test.float()
        sample_weights_test = class_weights_test[[t[1] for t in test_dataset.samples]]

        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights_train,
            num_samples=len(sample_weights_train),
            replacement=True,
        )

        self.val_sampler = WeightedRandomSampler(
            weights=sample_weights_val,
            num_samples=len(sample_weights_val),
            replacement=True,
        )

        self.test_sampler = WeightedRandomSampler(
            weights=sample_weights_test,
            num_samples=len(sample_weights_test),
            replacement=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=20,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=20,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=20,
        )


class HyenaModelPL(pl.LightningModule):
    def __init__(
        self,
        d_model,
        num_classes,
        width_max,
        height_max,
        num_epochs=10,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hyena_layer = HyenaOperator2D(
            d_model=d_model,
            width_max=width_max,
            height_max=height_max,
            order=order,
            filter_order=filter_order,
            dropout=dropout,
            filter_dropout=filter_dropout,
        )

        # apply max pooling to the output of the Hyena layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Fully connected layers
        self.linear1 = nn.Linear(d_model, 1024)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(128, num_classes)

        # Metrics
        self.train_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")

        self.train_f1 = F1Score(num_classes=num_classes, task="multiclass")
        self.val_f1 = F1Score(num_classes=num_classes, task="multiclass")
        self.test_f1 = F1Score(num_classes=num_classes, task="multiclass")

        self.train_auroc = AUROC(num_classes=num_classes, task="multiclass")
        self.val_auroc = AUROC(num_classes=num_classes, task="multiclass")
        self.test_auroc = AUROC(num_classes=num_classes, task="multiclass")

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):

        # what is the height and width of x?
        height_x, width__x = x.shape[-2], x.shape[-1]
        x = self.hyena_layer(x)

        print(x.shape)

        import sys

        sys.exit()

        # assert that x has shape (batch_size, d_model, height_max, width_max)
        assert (
            x.shape[-2] == self.hparams.height_max
        ), f"{x.shape[-2]} != {self.hparams.height_max}, which means the height of x is not correct"
        assert (
            x.shape[-1] == self.hparams.width_max
        ), f"{x.shape[-1]} != {self.hparams.width_max}, which means the width of x is not correct"
        assert (
            x.shape[1] == self.hparams.d_model
        ), f"{x.shape[1]} != {self.hparams.d_model}, which means the depth of x is not correct"

        x = self.maxpool(x)  # now x has shape (batch_size, d_model, 1, 1)

        # before passing x through the fully connected layers, we need to flatten it to shape (batch_size, d_model)
        x = torch.flatten(x, 1)

        assert (
            x.shape[1] == self.hparams.d_model and len(x.shape) == 2
        ), f"Shape of x is {x.shape}, should be (batch_size, d_model)"

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(logits, y))
        self.log("train_f1", self.train_f1(logits, y))
        self.log("train_auroc", self.train_auroc(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        self.log("val_accuracy", self.val_accuracy(logits, y))
        self.log("val_f1", self.val_f1(logits, y))
        self.log("val_auroc", self.val_auroc(logits, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", self.test_accuracy(logits, y))
        self.log("test_f1", self.test_f1(logits, y))
        self.log("test_auroc", self.test_auroc(logits, y))

    def on_train_epoch_end(self):
        # Log the current learning rate
        for scheduler in self.lr_schedulers():
            current_lr = scheduler.get_last_lr()[0]
            self.log("lr", current_lr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.num_epochs, eta_min=0
        )
        return [optimizer], [scheduler]


# Main training loop
def train_model(data_dir, num_gpus=3, num_epochs=10):
    data_module = ImageDataModule(
        data_dir=data_dir, batch_size=batch_size, downsample_factor=downsample_factor
    )

    model = HyenaModelPL(
        d_model=3,
        num_classes=23,
        num_epochs=10,
        width_max=96,
        height_max=96,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="hyena")

    # Trainer configuration for distributed training
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        devices=num_gpus,
        accelerator="gpu",  # 'ddp' for DistributedDataParallel
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())


if __name__ == "__main__":
    data_dir = "/media/hdd1/neo/pooled_deepheme_data"
    train_model(data_dir=data_dir)
