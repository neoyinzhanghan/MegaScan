import os
import pandas as pd
import openslide
import random
from pathlib import Path
from tqdm import tqdm

cellname = "PL1"

cell_data_path = "/media/ssd2/clinical_text_data/MegakaryoctePltClumpProject/slides_with_pl_cells_renamed.csv"

slide_folder = "/dmpisilon_tools/Greg/SF_Data/Pathology Images"

save_dir = "/media/hdd3/neo/PL1_cell_scan_training_data"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_regions_per_cell = 5
region_size = 512

df = pd.read_csv(cell_data_path)

metadata = {
    "data_idx": [],  # index of the region
    "slide_path": [],  # path to the slide
    "center_x": [],  # center x of the cell
    "center_y": [],  # center y of the cell
    "cellname": [],  # cellname
    "cell_image_size": [],  # size of the cell image (all sizes here are assumed to be at level 0 of the slide)
    "region_TL_x": [],  # top left x of the region
    "region_TL_y": [],  # top left y of the region
    "region_BR_x": [],  # bottom right x of the region
    "region_BR_y": [],  # bottom right y of the region
    "region_size": [],  # size of the region
    "center_x_rel": [],  # relative to the region
    "center_y_rel": [],  # relative to the region
}

current_index = 0

problem_slides = []


def find_file_recursive(slide_folder, slide_name):
    slide_folder_path = Path(slide_folder)
    for file_path in slide_folder_path.rglob(slide_name):
        return file_path
    return None


# traverse through rows in the dataframe
for i, row in tqdm(df.iterrows(), desc="Processing Cell Instances"):

    if row["cell_type"] != cellname:
        continue

    slide_name = row["slide_name"]

    # recursively search for the slide in the slide_folder with the slide_name
    slide_path = find_file_recursive(slide_folder, slide_name)

    if slide_path is None:
        # add the slide name to the problem slides list
        problem_slides.append(slide_name)

        continue

    center_x = row["center_x_slide"]
    center_y = row["center_y_slide"]
    cell_image_size = 256

    # calculate the range for the top left corner of the region
    min_TL_x = int(center_x - (region_size - cell_image_size // 2))
    max_TL_x = int(center_x - cell_image_size // 2)
    min_TL_y = int(center_y - (region_size - cell_image_size // 2))
    max_TL_y = int(center_y - cell_image_size // 2)

    for _ in range(num_regions_per_cell):

        try:
            # uniformly sample a top left corner
            region_TL_x = random.randint(min_TL_x, max_TL_x)
            region_TL_y = random.randint(min_TL_y, max_TL_y)

        except Exception as e:
            print("min_TL_x", min_TL_x)
            print("max_TL_x", max_TL_x)
            print("min_TL_y", min_TL_y)
            print("max_TL_y", max_TL_y)

            raise e

        region_BR_x = region_TL_x + region_size
        region_BR_y = region_TL_y + region_size

        center_x_rel = center_x - region_TL_x
        center_y_rel = center_y - region_TL_y

        metadata["data_idx"].append(current_index)
        metadata["slide_path"].append(slide_path)
        metadata["center_x"].append(center_x)
        metadata["center_y"].append(center_y)
        metadata["cellname"].append(cellname)
        metadata["cell_image_size"].append(cell_image_size)
        metadata["region_TL_x"].append(region_TL_x)
        metadata["region_TL_y"].append(region_TL_y)
        metadata["region_BR_x"].append(region_BR_x)
        metadata["region_BR_y"].append(region_BR_y)
        metadata["region_size"].append(region_size)
        metadata["center_x_rel"].append(center_x_rel)
        metadata["center_y_rel"].append(center_y_rel)

        # crop out the region using the openslide library at level 0 based on the computed coordinates
        try:
            slide = openslide.OpenSlide(slide_path)

            region = slide.read_region(
                (region_TL_x, region_TL_y), 0, (region_size, region_size)
            )

            slide.close()

        except Exception as e:

            # in the case of keyboard interrupt, just raise the exception
            if isinstance(e, KeyboardInterrupt):
                raise e

            print(f"Problem with slide {slide_path}")
            problem_slides.append(slide_path)
            print(e)
            continue

        # if the image is RGBA, convert it to RGB
        if region.mode == "RGBA":
            region = region.convert("RGB")

        # save the region as a jpg file in the save_dir with the name as the current_index.jpg
        region.save(os.path.join(save_dir, f"{current_index}.jpg"))

        current_index += 1

# convert the dictionary to a pandas dataframe
metadata_df = pd.DataFrame(metadata)

# save the metadata dataframe as a csv file in the save_dir with name metadata.csv
metadata_df.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

# save the list of problem slides as a txt file in the save_dir with name problem_slides.txt
with open(os.path.join(save_dir, "problem_slides.txt"), "w") as f:
    for slide in problem_slides:
        f.write(f"{slide}\n")
