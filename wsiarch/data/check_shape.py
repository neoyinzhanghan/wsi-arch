import os
import h5py
import pandas as pd
from tqdm import tqdm

data_dir = "/media/hdd1/neo/LUAD-LUSC_FI"

# get the path to all the h5 files in the data directory
h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
h5_paths = [os.path.join(data_dir, f) for f in h5_files]

shape_df = {
    "h5_path": [],
    "feature_image_shape": [],
}

for h5_path in tqdm(h5_paths, desc="Checking shapes..."):
    h5_file = h5py.File(h5_path, "r")
    feature_image_shape = h5_file["feature_image"][:].shape

    shape_df["h5_path"].append(h5_path)
    shape_df["feature_image_shape"].append(feature_image_shape)

shape_df = pd.DataFrame(shape_df)

# save the shape dataframe in the data directory
shape_df.to_csv(os.path.join(data_dir, "shape.csv"), index=False)
