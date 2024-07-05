import os
import shutil
import pandas as pd
from tqdm import tqdm

data_dir = "/media/hdd1/neo/LUAD-LUSC_FI_ResNet"
save_dir = "/media/hdd1/neo/LUAD-LUSC_FI_ResNet_lite"

# create the save directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_for_train_per_class = 10
num_for_val_per_class = 2
num_for_test_per_class = 2

# first open the metadata.csv file in the data directory, with columns idx,original_file,class,split
metadata_path = os.path.join(data_dir, "metadata.csv")

# read the metadata file
metadata = pd.read_csv(metadata_path)

# sample the metadata file
train_metadata = metadata[metadata["split"] == "train"]
val_metadata = metadata[metadata["split"] == "val"]
test_metadata = metadata[metadata["split"] == "test"]

# sample the metadata file randomly withouth replacement for each split based on num_for_train_per_class, num_for_val_per_class, num_for_test_per_class
train_metadata_sampled = train_metadata.groupby("class").sample(
    n=num_for_train_per_class, replace=False
)
val_metadata_sampled = val_metadata.groupby("class").sample(
    n=num_for_val_per_class, replace=False
)
test_metadata_sampled = test_metadata.groupby("class").sample(
    n=num_for_test_per_class, replace=False
)

# now put these into a new metadata file
new_metadata = pd.concat(
    [train_metadata_sampled, val_metadata_sampled, test_metadata_sampled]
)

# save the new metadata file in the save directory
new_metadata.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

# traverse through the rows of the new metadata file and copy the h5 files to the save directory
for idx, row in tqdm(new_metadata.iterrows(), desc="Copying files..."):
    # the idx column with .h5 is the file name of the h5 file in the data directory
    # copy that file to the save directory
    h5_file_path = str(row["idx"]) + ".h5"
    shutil.copy(
        os.path.join(data_dir, h5_file_path), os.path.join(save_dir, h5_file_path)
    )
