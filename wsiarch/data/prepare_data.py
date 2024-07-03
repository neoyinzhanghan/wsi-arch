import os
import csv
import random
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from wsiarch.data.load_h5 import h5_to_standard_format
from wsiarch.data.wsi_utils import get_thumbnail
from torchvision.transforms import ToTensor


def find_wsi_path(wsi_paths, h5_file_path):
    """
    Find the corresponding wsi path for the h5 file.
    """
    for wsi_path in wsi_paths:
        if os.path.basename(wsi_path).split(".")[-2] in h5_file_path:
            return wsi_path
    return None


def folder_as_class(folders, wsi_dirs, save_dir, train_prop=0.8):
    """
    Each folder is treated as a class. All the h5 files in the folder are data points for that class.
    First pool all the h5 files and then split them into train and validation sets.
    For each h5 file, give it an integer idx, and use h5_to_standard_format to get the triple (coords, pix_coords, feature_image).
    Find the corresponding wsi-path from the wsi_dirs and use get_thumbnail to get the thumbnail image, add that to the triple.
    The package the four tensors into a h5 file and save it to save_dir/idx.h5.
    Should save them as torch tensors to save_dir/idx/coords.pt, pix_coords.pt, feature_image.pt.
    Save metadata in a CSV file to keep track of idx, original h5 file name, class, and split.
    """
    idx = 0
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    metadata_path = save_dir_path / "metadata.csv"

    wsi_paths = []

    # pool all the .svs files in the wsi_dirs
    for wsi_dir in wsi_dirs:
        wsi_paths.extend(
            [
                os.path.join(wsi_dir, f)
                for f in os.listdir(wsi_dir)
                if f.endswith(".svs")
            ]
        )

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

                # Get the thumbnail
                wsi_path = find_wsi_path(wsi_paths, h5_file_path)

                assert (
                    wsi_path is not None
                ), f"Could not find wsi path for {h5_file_path}"
                thumbnail = get_thumbnail(wsi_path)

                thumbnail_tensor = ToTensor()(thumbnail)
                # Create the h5 file
                idx_h5_path = save_dir_path / f"{idx}.h5"

                with h5py.File(idx_h5_path, "w") as idx_h5_file:
                    idx_h5_file.create_dataset("coords", data=coords)
                    idx_h5_file.create_dataset("pix_coords", data=pix_coords)
                    idx_h5_file.create_dataset("feature_image", data=feature_image)
                    idx_h5_file.create_dataset("thumbnail", data=thumbnail_tensor)

                # Record metadata
                writer.writerow([idx, h5_file_path, class_name, split])
                idx += 1


if __name__ == "__main__":
    # folder_as_class(
    #     ["/Users/neo/Documents/MODS/wsiarch/examples"],
    #     "/Users/neo/Documents/MODS/wsiarch/tmp",
    # )

    # folder_as_class(
    #     folders=[
    #         "/media/hdd1/neo/TCGA-LUAD_SimCLR_2024-03-14",
    #         "/media/hdd1/neo/TCGA-LUSC_SimCLR_2024-04-08",
    #     ],
    #     wsi_dirs=["/media/ssd1/TCGA_WSI/TCGA-LUAD", "/media/ssd1/TCGA_WSI/TCGA-LUSC"],
    #     save_dir="/media/hdd1/neo/LUAD-LUSC_FI",
    # )

    # folder_as_class(
    #     folders=[
    #         "/media/hdd1/neo/TCGA-LUAD_SimCLR_2024-03-14",
    #         "/media/hdd1/neo/TCGA-LUSC_SimCLR_2024-04-08",
    #     ],
    #     wsi_dirs=["/media/ssd1/TCGA_WSI/TCGA-LUAD", "/media/ssd1/TCGA_WSI/TCGA-LUSC"],
    #     save_dir="/media/hdd1/neo/LUAD-LUSC_FI",
    # )

    folder_as_class(
        folders=[
            "/media/hdd1/neo/TCGA-LUSC_UNI",
            "/media/hdd1/neo/TCGA-LUAD_UNI",
        ],
        wsi_dirs=["/media/ssd1/TCGA_WSI/TCGA-LUAD", "/media/ssd1/TCGA_WSI/TCGA-LUSC"],
        save_dir="/media/hdd1/neo/LUAD-LUSC_FI_UNI",
    )