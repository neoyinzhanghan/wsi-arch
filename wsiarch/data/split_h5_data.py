import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the directories and save path
data_dir_dct = {
    "LUAD": "/media/hdd1/neo/TCGA-LUAD_UNI",
    "LUSC": "/media/hdd1/neo/TCGA-LUSC_UNI",
}

save_path = "/media/hdd1/neo/LUAD-LUSC_UNI_metadata.csv"

# Define the proportions for train, validation, and test splits
train_prop, val_prop, test_prop = 0.7, 0.2, 0.1

metadata = {
    "h5_path": [],
    "class": [],
    "split": [],
}

for class_name, data_dir in data_dir_dct.items():
    print(f"Processing {class_name} data...")

    h5_files = os.listdir(data_dir)
    h5_files = [f for f in h5_files if f.endswith(".h5")]
    h5_paths = [os.path.join(data_dir, f) for f in h5_files]

    # Split the data into train, validation, and test sets
    train_paths, test_paths = train_test_split(
        h5_paths, test_size=test_prop, random_state=42
    )
    train_paths, val_paths = train_test_split(
        train_paths, test_size=val_prop / (train_prop + val_prop), random_state=42
    )

    for split, paths in zip(
        ["train", "val", "test"], [train_paths, val_paths, test_paths]
    ):
        metadata["h5_path"].extend(paths)
        metadata["class"].extend([class_name] * len(paths))
        metadata["split"].extend([split] * len(paths))

metadata_df = pd.DataFrame(metadata)

metadata_df.to_csv(save_path, index=False)
