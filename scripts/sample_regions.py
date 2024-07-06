import os
import random
import shutil
import pandas as pd
from tqdm import tqdm

data_dir = "/media/hdd3/neo/results_bma_v4"
save_dir = "/media/hdd3/neo/PL1_cell_scan_training_data_non_pl1"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get all the subfolders that do not start with ERROR
subfolders = [f for f in os.listdir(data_dir) if not f.startswith("ERROR")]

# Make sure to check if they are actually folders, not files
subfolders = [f for f in subfolders if os.path.isdir(os.path.join(data_dir, f))]

for subfolder in tqdm(subfolders, desc="Sanity Checking Subfolders"):
    subfolder_path = os.path.join(data_dir, subfolder)

    # Check for the 'focus_regions' subfolder
    focus_regions_path = os.path.join(subfolder_path, "focus_regions")
    if not os.path.isdir(focus_regions_path):
        shutil.rmtree(subfolder_path)
        continue

    # Check for the 'high_mag_unannotated' subfolder within 'focus_regions'
    high_mag_unannotated_path = os.path.join(focus_regions_path, "high_mag_unannotated")
    if not os.path.isdir(high_mag_unannotated_path):
        shutil.rmtree(subfolder_path)

print("Sanity check completed.")

num_regions = 3000

current_idx = 0

metadata = {
    "idx": [],
    "slide_folder_path": [],
    "region_image_path": [],
    "region_idx": [],
}

for i in tqdm(range(num_regions), desc="Sampling Regions"):
    # randomly select a subfolder
    subfolder = random.choice(subfolders)

    # randomly select an image .jpg from the subfolder/focus_regions/high_mag_unannotated
    image_folder = os.path.join(
        data_dir, subfolder, "focus_regions", "high_mag_unannotated"
    )

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    image_file = random.choice(image_files)

    # the region_idx is the raw filename without the extension
    region_idx = os.path.splitext(image_file)[0]
    # make sure you have an integer index
    region_idx = int(region_idx)

    region_image_path = os.path.join(image_folder, image_file)

    # copy the image to the save_dir
    shutil.copy(region_image_path, save_dir)

    # add the information of the region image to the metadata
    metadata["idx"].append(current_idx)
    metadata["slide_folder_path"].append(subfolder)
    metadata["region_image_path"].append(region_image_path)
    metadata["region_idx"].append(region_idx)

    current_idx += 1

# convert the metadata to a pandas dataframe
df = pd.DataFrame(metadata)

# save the metadata to a csv file
df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
