import os
import csv
import random
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from wsiarch.data.load_h5 import h5_to_standard_format


def folder_as_class(folders, save_dir, train_prop=0.8):
    """
    Each folder is treated as a class. All the h5 files in the folder are data points for that class.
    First pool all the h5 files and then split them into train and validation sets.
    For each h5 file, give it an integer idx, and use h5_to_standard_format to get the triple (coords, pix_coords, feature_image).
    Should save them as torch tensors to save_dir/idx/coords.pt, pix_coords.pt, feature_image.pt.
    Save metadata in a CSV file to keep track of idx, original h5 file name, class, and split.
    """
    idx = 0
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    metadata_path = save_dir_path / "metadata.csv"

    with open(metadata_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "original_file", "class", "split"])

        for folder in folders:
            class_name = os.path.basename(folder)
            h5_files = [f for f in os.listdir(folder) if f.endswith(".h5")]
            random.shuffle(h5_files)
            split_idx = int(len(h5_files) * train_prop)
            train_files, val_files = h5_files[:split_idx], h5_files[split_idx:]

            for h5_file_path, split in tqdm(
                zip(
                    h5_files,
                    ["train"] * split_idx + ["val"] * (len(h5_files) - split_idx),
                ),
                total=len(h5_files),
                desc=f"Processing {class_name}...",
            ):

                file_path = os.path.join(folder, h5_file_path)

                # open the h5 file
                h5_file = h5py.File(file_path, "r")
                coords, pix_coords, feature_image = h5_to_standard_format(h5_file)

                # Create subdirectory for each data point
                idx_dir = save_dir_path / str(idx)
                idx_dir.mkdir()

                # Save data
                torch.save(coords, idx_dir / "coords.pt")
                torch.save(pix_coords, idx_dir / "pix_coords.pt")
                torch.save(feature_image, idx_dir / "feature_image.pt")

                # Record metadata
                writer.writerow([idx, h5_file_path, class_name, split])
                idx += 1


if __name__ == "__main__":
    folder_as_class(
        ["/media/hdd1/neo/TCGA-LUAD_SimCLR_2024-03-14", "/media/hdd1/neo/TCGA-LUSC_SimCLR_2024-04-08"],
        "/media/hdd1/neo/LUAD-LUSC_FI",
    )
