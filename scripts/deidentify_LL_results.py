import os
from tqdm import tqdm

results_dir = "/media/hdd1/neo/results_bma_aml_v3_cleaned"

# get the path of all the subdirectories in the results directory
subdirs = [
    os.path.join(results_dir, o)
    for o in os.listdir(results_dir)
    if os.path.isdir(os.path.join(results_dir, o))
]


current_idx = 0
# iterate over all the subdirectories
for subdir in tqdm(subdirs, desc="Deidentifying results folders"):
    # rename the subdirectory to the current index
    os.rename(subdir, os.path.join(results_dir, str(current_idx)))

    current_idx += 1
