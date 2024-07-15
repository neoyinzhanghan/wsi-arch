# import torch
import os
import pandas as pd
import h5py
import pytorch_lightning as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as Dataloader


cellnames = [
    "B1",
    "B2",
    "E1",
    "E4",
    "ER1",
    "ER2",
    "ER3",
    "ER4",
    # "ER5",
    # "ER6",
    "L2",
    "L4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "MO2",
    # "PL2",
    # "PL3",
    # "U1",
    # "U4",
]


class HemeCellMILDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split="train",
        length_max=100,
        feature_name="features_v3",
        transform=None,
    ):  # the choice of 58182 here is based on the maximum context length required for the LUAD vs LUSC problem
        """
        Args:
            metadata_path (string): Path to the metadata csv file.
            split (string): One of 'train' or 'val' to specify which split to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.transform = transform
        self.length_max = length_max
        self.all_classes = self.metadata["label"].unique()
        self.feature_name = feature_name

        # Create a mapping from class names to indices
        self.class_to_index = {
            cls: idx for idx, cls in enumerate(self.metadata["label"].unique())
        }
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        # first randomly select a label
        class_label = np.random.choice(self.all_classes)

        # get the metadata for the selected class
        metadata = self.metadata[self.metadata["label"] == class_label]

        # randomly select a row from the metadata
        metadata_row = metadata.sample()

        slide_result_path = metadata_row["slide_result_path"].values[0]

        all_feature_paths = []
        for cellname in cellnames:
            feature_dir = os.path.join(
                slide_result_path, "cells", cellname, self.feature_name
            )

            # check if is dir
            if os.path.isdir(feature_dir):
                feature_files = os.listdir(feature_dir)
                # check that the features end with .pt
                feature_files = [f for f in feature_files if f.endswith(".pt")]
                feature_paths = [os.path.join(feature_dir, f) for f in feature_files]

                all_feature_paths.extend(feature_paths)

        # randomly select self.length_max number of features, if there are less than self.length_max features, use bootstrapping
        if len(all_feature_paths) < self.length_max:
            feature_paths = np.random.choice(
                all_feature_paths, self.length_max, replace=True
            )
        else:
            feature_paths = np.random.choice(all_feature_paths, self.length_max)

        # each features has a shape of [d,], stack them to [self.length_max, d]
        features = []

        for feature_path in feature_paths:
            feature_path = os.path.join(feature_dir, feature_path)
            feature = torch.load(feature_path)
            features.append(feature)

        x = torch.stack(features)

        # Get the class label
        class_label = self.metadata.iloc[idx]["label"]
        class_index = self.class_to_index[class_label]

        return x, class_index


def create_data_loaders(
    metadata_path,
    length_max=100,
    feature_name="features_v3",
    batch_size=32,
    num_workers=12,
):

    train_dataset = HemeCellMILDataset(
        metadata_path=metadata_path,
        split="train",
        length_max=length_max,
        feature_name=feature_name,
    )

    val_dataset = HemeCellMILDataset(
        metadata_path=metadata_path,
        split="val",
        length_max=length_max,
        feature_name=feature_name,
    )

    test_dataset = HemeCellMILDataset(
        metadata_path=metadata_path,
        split="test",
        length_max=length_max,
        feature_name=feature_name,
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


class HemeCellMILModule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path,
        length_max=100,
        feature_name="features_v3",
        batch_size=32,
        num_workers=12,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.length_max = length_max
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_name = feature_name

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            metadata_path=self.metadata_path,
            length_max=self.length_max,
            batch_size=self.batch_size,
            feature_name=self.feature_name,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
