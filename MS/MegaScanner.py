####################################################################################################
# Imports ##########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import os
import ray
import openslide
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import sys
import shutil
from PIL import Image
from tqdm import tqdm
from ray.exceptions import RayTaskError
from pathlib import Path

# Within package imports ###########################################################################
from MS.PBDifferential import Differential, to_count_dict
from MS.FileNameManager import FileNameManager
from MS.BMATopView import TopView, SpecimenError, TopViewError
from MS.brain.HemeLabelLightningManager import HemeLabelLightningManager
from MS.brain.CellScannerManager import CellScannerManager
from MS.brain.YOLOManager import YOLOManager
from MS.brain.FeatureEngineer import CellFeatureEngineer
from MS.vision.processing import SlideError, read_with_timeout
from MS.vision.BMAWSICropManager import WSICropManager
from MS.communication.write_config import *
from MS.communication.visualization import *
from MS.brain.utils import *
from MS.communication.saving import *
from MS.brain.SpecimenClf import get_specimen_type
from MS.SearchView import SearchView
from MS.BMAFocusRegion import *
from MS.BMAFocusRegionTracker import FocusRegionsTracker, NotEnoughFocusRegionsError
from MS.brain.BMAHighMagRegionCheckTracker import BMAHighMagRegionCheckTracker
from MS.resources.BMAassumptions import *


class MegaScanner:
    """A Class representing the scanning process of a WSI for megakaryocytes.

    === Class Attributes ===
    - file_name_manager : the FileNameManager class object of the WSI
    - top_view : the TopView class object of the WSI
    - wbc_candidates : a list of WBCCandidate class objects that represent candidates for being WBCs
    - focus_regions : a list of FocusRegion class objects representing the focus regions of the search view
    - differential : a Differential class object representing the differential of the WSI
    - save_dir : the directory to save the diagnostic logs, dataframes (and images if hoarding is True
    - profiling_data : a dictionary containing the profiling data of the PBCounter object

    - verbose : whether to print out the progress of the PBCounter object
    - hoarding : whether to hoard regions and cell images processed into permanent storage
    - continue_on_error : whether to continue processing the WSI if an error occurs
    - ignore_specimen_type : whether to ignore the specimen type of the WSI
    - do_extract_features : whether to extract features from the WBC candidates
    - fr_tracker : a FocusRegionsTracker object that tracks the focus regions

    - predicted_specimen_type: the predicted specimen type of the WSI
    - wsi_path : the path to the WSI
    """

    def __init__(
        self,
        wsi_path: str,
        verbose: bool = False,
        hoarding: bool = False,
        extra_hoarding: bool = False,
        continue_on_error: bool = False,
        ignore_specimen_type: bool = False,
        do_extract_features: bool = False,
        overwrite: bool = True,
        error: bool = False,
    ):
        """Initialize a PBCounter object."""

        self.profiling_data = {}

        start_time = time.time()

        self.verbose = verbose
        self.hoarding = hoarding
        self.continue_on_error = continue_on_error
        self.ignore_specimen_type = ignore_specimen_type
        self.wsi_path = wsi_path
        self.do_extract_features = do_extract_features
        self.overwrite = overwrite
        self.error = error
        self.extra_hoarding = extra_hoarding

        # The focus regions and WBC candidates are None until they are processed
        self.focus_regions = None
        self.wbc_candidates = None
        self.fr_tracker = None
        self.differential = None

        if self.verbose:
            print(f"Initializing FileNameManager object for {wsi_path}")
        # Initialize the manager
        self.file_name_manager = FileNameManager(wsi_path)

        self.save_dir = os.path.join(dump_dir, self.file_name_manager.stem)

        # if the save_dir already exists, then delete it and make a new one
        if os.path.exists(self.save_dir):
            if self.overwrite:
                os.system(f"rm -r '{self.save_dir}'")

        # if the save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            # Processing the WSI
            try:
                if self.verbose:
                    print(f"Opening WSI as {wsi_path}")

                wsi = openslide.OpenSlide(wsi_path)
            except Exception as e:
                self.error = True
                print(f"Error occurred: {e}")
                raise SlideError(e)

            if self.verbose:
                print(f"Processing WSI top view as TopView object")
            # Processing the top level image
            top_level = topview_level

            if self.verbose:
                print(f"Obtaining top view image")

            # if the read_region takes longer than 10 seconds, then raise a SlideError

            top_view = read_with_timeout(
                wsi, (0, 0), top_level, wsi.level_dimensions[top_level]
            )

            print("Checking Specimen Type")
            specimen_type = get_specimen_type(top_view)

            self.predicted_specimen_type = specimen_type

            if specimen_type != "Bone Marrow Aspirate":
                if not self.ignore_specimen_type:
                    # if self.continue_on_error:

                    #     e = SpecimenError(
                    #         "The specimen is not Bone Marrow Aspirate. Instead, it is "
                    #         + specimen_type
                    #         + "."
                    #     )

                    #     print(
                    #         "ERROR: The specimen is not Bone Marrow Aspirate. Instead, it is "
                    #         + specimen_type
                    #         + "."
                    #     )

                    #     # if the save_dir does not exist, create it
                    #     os.makedirs(self.save_dir, exist_ok=True)

                    #     # save the exception and profiling data
                    #     with open(os.path.join(self.save_dir, "error.txt"), "w") as f:
                    #         f.write(str(e))

                    #     # Save profiling data even in case of error
                    #     with open(
                    #         os.path.join(self.save_dir, "runtime_data.yaml"), "w"
                    #     ) as file:
                    #         yaml.dump(
                    #             self.profiling_data,
                    #             file,
                    #             default_flow_style=False,
                    #             sort_keys=False,
                    #         )

                    #     # rename the save_dir name to "ERROR_" + save_dir
                    #     os.rename(
                    #         self.save_dir,
                    #         os.path.join(
                    #             dump_dir,
                    #             "ERROR_" + Path(self.file_name_manager.wsi_path).stem,
                    #         ),
                    #     )

                    #     print(f"Error occurred and logged. Continuing to next WSI.")

                    # else:
                    raise SpecimenError(
                        "The specimen is not Bone Marrow Aspirate. Instead, it is "
                        + specimen_type
                        + "."
                    )

                else:
                    print(
                        "USERWarning: The specimen is not Bone Marrow Aspirate. Instead, it is "
                        + specimen_type
                        + "."
                    )

            # top_view = wsi.read_region(
            #     (0, 0), top_level, wsi.level_dimensions[top_level])

            top_view = top_view.convert("RGB")
            top_view_downsampling_rate = wsi.level_downsamples[top_level]

            try:
                self.top_view = TopView(
                    top_view,
                    top_view_downsampling_rate,
                    top_level,
                    verbose=self.verbose,
                )
            except Exception as e:
                self.error = True
                print(f"Error occurred: {e}")
                raise TopViewError(e)

            self.top_view.save_images(self.save_dir)

            if self.verbose:
                print(f"Processing WSI search view as SearchView object")
            # Processing the search level image

            if self.verbose:
                print(f"Closing WSI")
            wsi.close()

            self.profiling_data["init_time"] = time.time() - start_time

        except Exception as e:
            self.error = True
            if self.continue_on_error:
                print(f"Error occurred: {e}")
                print(f"Continuing to next WSI.")

                # remame the save_dir to "ERROR_" + save_dir
                error_path = os.path.join(
                    dump_dir, "ERROR_" + self.file_name_manager.stem
                )

                # if the error_path already exists, then delete it and make a new one
                if os.path.exists(error_path):
                    os.system(f"rm -r '{error_path}'")

                os.rename(
                    self.save_dir,
                    os.path.join(
                        dump_dir, "ERROR_" + Path(self.file_name_manager.wsi_path).stem
                    ),
                )

                self.error = True

                # save the exception message as a txt file in the ERROR_save_dir
                with open(os.path.join(error_path, "error.txt"), "w") as f:
                    f.write(str(e))

            else:
                self.error = True
                raise e

    def find_focus_regions(self):
        """Return the focus regions of the highest magnification view."""

        start_time = time.time()

        os.makedirs(os.path.join(self.save_dir, "focus_regions"), exist_ok=True)

        # First get a list of the focus regions coordinates based on focus_regions_size at highest level of magnification
        # if the dimension is not divisible by the focus_regions_size, then we simply omit the last focus region

        # get the dimension of the highest mag, which is the level 0
        # get the level 0 dimension using
        wsi = openslide.OpenSlide(self.wsi_path)
        search_view_dimension = wsi.level_dimensions[search_view_level]
        wsi.close()

        dimx, dimy = search_view_dimension

        # get the number of focus regions in the x and y direction
        num_focus_regions_x = dimx // search_view_focus_regions_size
        num_focus_regions_y = dimy // search_view_focus_regions_size

        # get the list of focus regions coordinates
        focus_regions_coordinates = []

        for i in range(num_focus_regions_x):
            for j in range(num_focus_regions_y):
                focus_regions_coordinates.append(
                    (
                        i * search_view_focus_regions_size,
                        j * search_view_focus_regions_size,
                        (i + 1) * search_view_focus_regions_size,
                        (j + 1) * search_view_focus_regions_size,
                    )
                )

        focus_regions_coordinates = self.top_view.filter_coordinates_with_mask(
            focus_regions_coordinates
        )

        # # take the 300 focus regions from the middle of the list which is len(focus_regions_coordinates) // 2 - 150 to len(focus_regions_coordinates) // 2 + 150
        # half = 300 // 2
        # focus_regions_coordinates = focus_regions_coordinates[
        #     len(focus_regions_coordinates) // 2
        #     - half : len(focus_regions_coordinates) // 2
        #     + half
        # ]

        ray.shutdown()
        # ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        ray.init()

        list_of_batches = create_list_of_batches_from_list(
            focus_regions_coordinates, region_cropping_batch_size
        )

        if self.verbose:
            print("Initializing WSICropManager")
        task_managers = [
            WSICropManager.remote(self.wsi_path) for _ in range(num_croppers)
        ]

        tasks = {}
        all_results = []

        for i, batch in enumerate(list_of_batches):
            manager = task_managers[i % num_labellers]
            task = manager.async_get_bma_focus_region_batch.remote(batch)
            tasks[task] = batch

        with tqdm(
            total=len(focus_regions_coordinates), desc="Cropping focus regions"
        ) as pbar:
            while tasks:
                done_ids, _ = ray.wait(list(tasks.keys()))

                for done_id in done_ids:
                    try:
                        batch = ray.get(done_id)
                        for focus_region in batch:
                            all_results.append(focus_region)

                            pbar.update()

                    except RayTaskError as e:
                        self.error = True
                        print(
                            f"Task for focus region {tasks[done_id]} failed with error: {e}"
                        )

                    del tasks[done_id]

        if self.verbose:
            print(f"Shutting down Ray")
        ray.shutdown()

        self.focus_regions = all_results

        self.profiling_data["cropping_focus_regions_time"] = time.time() - start_time

    def filter_focus_regions(self):
        start_time = time.time()

        fr_tracker = FocusRegionsTracker(self.focus_regions)

        self.fr_tracker = fr_tracker

        self.fr_tracker.compute_resnet_confidence()
        selected_focus_regions = self.fr_tracker.get_top_n_focus_regions()
        self.fr_tracker.save_results(self.save_dir)

        self.focus_regions = selected_focus_regions

        self.profiling_data["filtering_focus_regions_time"] = time.time() - start_time

        if (
            self.extra_hoarding
        ):  # if extra hoarding is True, then save the focus regions
            start_time = time.time()
            self.fr_tracker.save_all_focus_regions(self.save_dir)
            self.profiling_data["hoarding_focus_regions_time"] = (
                time.time() - start_time
            )
        else:
            self.profiling_data["hoarding_focus_regions_time"] = 0

        # now for each focus region, we will find get the image

        start_time = time.time()

        for focus_region in tqdm(
            self.focus_regions, desc="Getting high magnification focus region images"
        ):
            wsi = openslide.OpenSlide(self.wsi_path)

            pad_size = snap_shot_size // 2

            padded_coordinate = (
                focus_region.coordinate[0] - pad_size,
                focus_region.coordinate[1] - pad_size,
                focus_region.coordinate[2] + pad_size,
                focus_region.coordinate[3] + pad_size,
            )
            padded_image = wsi.read_region(
                padded_coordinate,
                0,
                (
                    focus_region.coordinate[2]
                    - focus_region.coordinate[0]
                    + pad_size * 2,
                    focus_region.coordinate[3]
                    - focus_region.coordinate[1]
                    + pad_size * 2,
                ),
            )

            original_width = focus_region.coordinate[2] - focus_region.coordinate[0]
            original_height = focus_region.coordinate[3] - focus_region.coordinate[1]

            unpadded_image = padded_image.crop(
                (
                    pad_size,
                    pad_size,
                    pad_size + original_width,
                    pad_size + original_height,
                )
            )

            focus_region.get_image(unpadded_image, padded_image)

        self.profiling_data["getting_high_mag_images_time"] = time.time() - start_time

        start_time = time.time()

        high_mag_check_tracker = BMAHighMagRegionCheckTracker(
            focus_regions=self.focus_regions
        )

        good_focus_regions = high_mag_check_tracker.get_good_focus_regions()

        self.focus_regions = good_focus_regions

        high_mag_check_tracker.save_results(self.save_dir)

        self.profiling_data["high_mag_check_time"] = time.time() - start_time

        if self.hoarding:
            start_time = time.time()
            high_mag_check_tracker.hoard_results(self.save_dir)
            self.profiling_data["hoarding_high_mag_check_time"] = (
                time.time() - start_time
            )
        else:
            self.profiling_data["hoarding_high_mag_check_time"] = 0

    def scan_for_megakaryocytes(self):
        """Update the labels of the wbc_candidates of the PBCounter object."""

        start_time = time.time()

        ray.shutdown()
        # ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        ray.init()

        list_of_batches = create_list_of_batches_from_list(
            self.focus_regions, region_clf_batch_size
        )

        if self.verbose:
            print("Initializing CellScannerManagers")
        task_managers = [
            CellScannerManager.remote(cell_scanner_ckpt_path)
            for _ in range(num_labellers)
        ]

        tasks = {}
        all_results = []

        for i, batch in enumerate(list_of_batches):
            manager = task_managers[i % num_labellers]
            task = manager.async_scan_regions_batch.remote(batch)
            tasks[task] = batch

        with tqdm(
            total=len(self.focus_regions),
            desc="Scanning for Target Cells in Focus Regions",
        ) as pbar:
            while tasks:
                done_ids, _ = ray.wait(list(tasks.keys()))

                for done_id in done_ids:
                    try:
                        batch = ray.get(done_id)
                        for focus_region in batch:
                            all_results.append(focus_region)

                            pbar.update()

                    except RayTaskError as e:
                        self.error = True
                        print(
                            f"Task for focus region {tasks[done_id]} failed with error: {e}"
                        )

                    del tasks[done_id]

        if self.verbose:
            print(f"Shutting down Ray")
        ray.shutdown()

        self.focus_regions = all_results
        self.profiling_data["mega_scanner_time"] = time.time() - start_time
        
