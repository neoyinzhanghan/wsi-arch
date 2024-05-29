import torch
import os
import pandas as pd
import h5py
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as Dataloader


def random_up_padding(feature_image, width_max, height_max):
    # the feature image should have shape (depth, width, height) and height should be <= height_max, width should be <= width_max
    depth, width, height = feature_image.shape
    assert (
        width <= width_max and height <= height_max
    ), f"The width {width} and height {height} should be less than or equal to width_max {width_max} and height_max {height_max} respectively."

    # create a tensor of zeros with shape (depth, width_max, height_max) by randomly finding the top left corner to place the feature image, the rest of the tensor will be zeros\
    padded_feature_image = np.zeros((depth, width_max, height_max))

    # randomly find the top left corner to place the feature image
    top_left_x = np.random.randint(0, width_max - width)
    top_left_y = np.random.randint(0, height_max - height)

    # place the feature image in the padded_feature_image tensor
    padded_feature_image[
        :, top_left_x : top_left_x + width, top_left_y : top_left_y + height
    ] = feature_image

    return padded_feature_image


class FeatureImageDataset(Dataset):
    def __init__(
        self, root_dir, metadata_file, split, width_max, height_max, transform=None
    ):
        """
        Args:
            root_dir (string): Directory with all the h5 files.
            metadata_file ( string): Path to the metadata csv file.
            split (string): One of 'train' or 'val' to specify which split to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        metadata_path = os.path.join(root_dir, metadata_file)
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.width_max = width_max
        self.height_max = height_max

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = os.path.join(
            self.root_dir, str(self.metadata.iloc[idx]["idx"]) + ".h5"
        )

        h5_file = h5py.File(h5_path, "r")

        # get the "feature_image" dataset
        feature_image = h5_file["feature_image"][:]

        # randomly pad the feature image
        feature_image = random_up_padding(
            feature_image, width_max=self.width_max, height_max=self.height_max
        )

        if self.transform:
            sample = self.transform(feature_image)

        return sample


def create_data_loaders(
    root_dir, metadata_file, width_max, height_max, batch_size=32, num_workers=12
):

    train_dataset = FeatureImageDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split="train",
        width_max=width_max,
        height_max=height_max,
    )

    val_dataset = FeatureImageDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split="val",
        width_max=width_max,
        height_max=height_max,
    )

    test_dataset = FeatureImageDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split="test",
        width_max=width_max,
        height_max=height_max,
    )

    train_loader = Dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = Dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = Dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


class FeatureImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        metadata_file,
        width_max,
        height_max,
        batch_size=32,
        num_workers=12,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.metadata_file = metadata_file
        self.width_max = width_max
        self.height_max = height_max
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            root_dir=self.root_dir,
            metadata_file=self.metadata_file,
            width_max=self.width_max,
            height_max=self.height_max,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
