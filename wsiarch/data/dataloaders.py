import torch
import os
import pandas as pd
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as Dataloader


class FeatureImageDataset(Dataset):
    def __init__(self, root_dir, metadata_file, split, transform=None):
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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = os.path.join(self.root_dir, self.metadata.iloc[idx]["idx"] + ".h5")
        h5_file = h5py.File(h5_path, "r")

        # get the "feature_image" dataset
        feature_image = h5_file["feature_image"][:]

        if self.transform:
            sample = self.transform(feature_image)

        return sample


def create_data_loaders(root_dir, metadata_file, batch_size=32, num_workers=12):

    train_dataset = FeatureImageDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split="train",
    )

    val_dataset = FeatureImageDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split="val",
    )

    train_loader = Dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = Dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader
