import torch
import os
import pandas as pd
import h5py
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as Dataloader


class H5Dataset(Dataset):
    def __init__(
        self, metadata_path, split, length_max=58182, transform=None
    ):  # the choice of 58182 here is based on the maximum context length required for the LUAD vs LUSC problem
        """
        Args:
            root_dir (string): Directory with all the h5 files.
            metadata_file (string): Path to the metadata csv file.
            split (string): One of 'train' or 'val' to specify which split to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.transform = transform
        self.length_max = length_max

        # Create a mapping from class names to indices
        self.class_to_index = {
            cls: idx for idx, cls in enumerate(self.metadata["class"].unique())
        }
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = self.metadata.iloc[idx]["h5_path"]

        h5_file = h5py.File(h5_path, "r")

        # get the "features" dataset
        features = h5_file["features"][:]

        # get the "coords" data
        coords = h5_file["coords"][:]

        # features shape is [l, d] where l is the context length and d is the depth of the features
        # coords shape is [l, 2] where l is the context length

        # pad the features and coords to have length self.length_max, add zeros to the right
        x = np.pad(features, ((0, self.length_max - features.shape[0]), (0, 0)), mode="constant")
        p = np.pad(coords, ((0, self.length_max - coords.shape[0]), (0, 0)), mode="constant")

        x = torch.tensor(x, dtype=torch.float32)
        p = torch.tensor(p, dtype=torch.float32)

        # Get the class label
        class_label = self.metadata.iloc[idx]["class"]
        class_index = self.class_to_index[class_label]

        return x, p, class_index


def create_data_loaders(
    root_dir, metadata_path, length_max=58182, batch_size=32, num_workers=12
):

    train_dataset = H5Dataset(
        root_dir=root_dir,
        metadata_file=metadata_path,
        split="train",
        length_max=length_max,
    )

    val_dataset = H5Dataset(
        root_dir=root_dir,
        metadata_file=metadata_path,
        split="val",
        length_max=length_max,
    )

    test_dataset = H5Dataset(
        root_dir=root_dir,
        metadata_file=metadata_path,
        split="test",
        length_max=length_max,
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


class H5DataModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path,
        length_max=58182,
        batch_size=32,
        num_workers=12,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.length_max = length_max
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            metadata_path=self.metadata_path,
            length_max=self.length_max,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
