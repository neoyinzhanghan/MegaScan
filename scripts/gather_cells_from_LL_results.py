import os
import pandas as pd
from tqdm import tqdm


def grab_scan_data_from_LL_results(slides_dir, results_dir, cellname="B1"):
    """ """

    df_dct = {
        "slide_path": [],
        "center_x": [],
        "center_y": [],
        "cellname": [],
        "cell_image_size": [],
    }

    # first get all the folders in the results_dir that do not start with name ERROR_
    result_folders = [f for f in os.listdir(results_dir) if not f.startswith("ERROR_")]

    # make sure to check if they are actually folders not files
    result_folders = [
        f for f in result_folders if os.path.isdir(os.path.join(results_dir, f))
    ]

    # iterate over the result folders
    for result_folder in tqdm(result_folders, desc="Processing Result Folders"):

        # get the path which is the folder/cells/cells_info.csv
        cells_info_path = os.path.join(
            results_dir, result_folder, "cells", "cells_info.csv"
        )

        # open as a pandas dataframe
        cells_info = pd.read_csv(cells_info_path)

        # only get all the rows where the label column is equal to the cellname
        cells_info = cells_info[cells_info["label"] == cellname]

        # use the TL_x, TL_y, BR_x, BR_y to calculate the center_x and center_y
        cells_info["center_x"] = (cells_info["TL_x"] + cells_info["BR_x"]) / 2
        cells_info["center_y"] = (cells_info["TL_y"] + cells_info["BR_y"]) / 2

        # iterate over the rows in the cells_info dataframe
        for i, row in cells_info.iterrows():
            # get the slide path
            slide_path = os.path.join(slides_dir, result_folder, "slide.tif")

            # get the center_x, center_y, cellname, and cell_image_size
            center_x = row["center_x"]
            center_y = row["center_y"]
            cellname_label = row["label"]

            # append to the dictionary
            df_dct["slide_path"].append(slide_path)
            df_dct["center_x"].append(center_x)
            df_dct["center_y"].append(center_y)
            df_dct["cellname"].append(cellname_label)
            df_dct["cell_image_size"].append(96)

    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame(df_dct)

    return df


if __name__ == "__main__":
    slides_dir = "/media/hdd2/neo/BMA_Normal_lite"
    results_dir = "/media/hdd3/neo/results_bma_normal_lite_v3"

    df = grab_scan_data_from_LL_results(slides_dir, results_dir)

    df.to_csv(
        "/media/hdd3/neo/results_bma_normal_lite_v3/B1_cell_scan_training_data.csv",
        index=False,
    )
