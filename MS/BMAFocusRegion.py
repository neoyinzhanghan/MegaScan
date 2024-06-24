####################################################################################################
# Imports ###########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import os
import numpy as np
import pandas as pd
import ray
from PIL import Image
from PIL import Image

# Within package imports ###########################################################################
from MS.vision.image_quality import VoL, WMP
from MS.communication.visualization import annotate_focus_region, draw_dashed_rect
from MS.resources.BMAassumptions import *


class FocusRegion:
    """A focus region class object representing all the information needed at the focus region of the WSI.

    === Class Attributes ===
    - idx : the index of the focus region
    - coordinate : the coordinate of the focus region in the level 0 view in the format of (TL_x, TL_y, BR_x, BR_y)
    - image : the image of the focus region
    - annotated_image : the image of the focus region annotated with the WBC candidates
    - VoL : the variance of the laplacian of the focus region
    - wbc_candidate_bboxes : a list of bbox of the WBC candidates in the level 0 view in the format of (TL_x, TL_y, BR_x, BR_y) in relative to the focus region
    - wbc_candidates : a list of wbc_candidates objects
    - YOLO_df : should contain the good bounding boxes relative location to the focus region, the absolute coordinate of the focus region, and the confidence score of the bounding box
    - adequate_confidence_score : the confidence score of the region classification model
    - downsampled_coordinate : the coordinate of the focus region in the downsampled view in the format of (TL_x, TL_y, BR_x, BR_y)
    - downsampled_image : the downsampled image of the focus region
    - padded_image : the padded image of the focus region
    - image_mask_duo : one image where the downsampled image and mask are put side by side
    - VoL_high_mag : the variance of the laplacian of the high magnification image of the focus region
    - adequate_confidence_score_high_mag : the confidence score of the region classification model for the high magnification image
    """

    def __init__(self, downsampled_coordinate, downsampled_image, idx=None):
        """Initialize a FocusRegion object. The full resolution image is None at initialization."""

        self.idx = idx
        self.downsampled_coordinate = downsampled_coordinate
        self.coordinate = (
            downsampled_coordinate[0] * search_view_downsample_rate,
            downsampled_coordinate[1] * search_view_downsample_rate,
            downsampled_coordinate[2] * search_view_downsample_rate,
            downsampled_coordinate[3] * search_view_downsample_rate,
        )
        self.downsampled_image = downsampled_image
        self.image = None
        self.padded_image = None
        self.annotated_image = None

        # calculate the downsampled coordinateF

        # Calculate the VoL and WMP
        self.VoL = VoL(self.downsampled_image)
        # self.WMP, self.otsu_mask = WMP(self.image)　# for bone marrow aspirate we are not gonnae need this for now

        # image_mask_duo is one image where the downsampled image and mask are put side by side
        # note that mask is a black and white binary image while the downsampled image is a color image
        # so we need to convert the mask to a color image

        # Assuming self.downsampled_image is a PIL image, convert it to a NumPy array
        image_array = np.array(self.downsampled_image)

        # Convert RGBA to RGB if the alpha channel is not necessary
        if image_array.shape[2] == 4:
            image_array = image_array[
                :, :, :3
            ]  # This keeps the R, G, B channels and discards the alpha channel

        # Convert the binary mask to a 3-channel RGB image
        # otsu_rgb = np.stack((self.otsu_mask,) * 3, axis=-1)

        # Horizontally stack the two images
        # self.image_mask_duo = Image.fromarray(np.hstack((image_array, otsu_rgb)))

        self.adequate_confidence_score = None

        self.wbc_candidate_bboxes = None
        self.wbc_candidates = None
        self.YOLO_df = None

        VoL_high_mag = None
        adequate_confidence_score_high_mag = None

        self.mega_scan_softmax_vector = None

    def get_image(self, image, padded_image):
        """Update the image of the focus region."""

        # if the image is RGBA, convert it to RGB
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if padded_image.mode == "RGBA":
            padded_image = padded_image.convert("RGB")

        self.image = image
        self.padded_image = padded_image

    def get_annotated_image(self):
        """Return the image of the focus region annotated with the WBC candidates."""

        if self.image is None or self.wbc_candidate_bboxes is None:
            raise self.FocusRegionNotAnnotatedError

        elif self.annotated_image is not None:
            return self.annotated_image

        else:

            pad_size = snap_shot_size // 2

            padded_wbc_candidate_bboxes = [
                (
                    bbox[0] + pad_size,
                    bbox[1] + pad_size,
                    bbox[2] + pad_size,
                    bbox[3] + pad_size,
                )
                for bbox in self.wbc_candidate_bboxes
            ]

            self.annotated_image = annotate_focus_region(
                self.padded_image, padded_wbc_candidate_bboxes
            )

            # Calculate the coordinates for the green rectangle
            top_left = (pad_size, pad_size)
            bottom_right = (
                pad_size + focus_regions_size,
                pad_size + focus_regions_size,
            )

            # Draw the green rectangle
            draw_dashed_rect(
                self.annotated_image,
                top_left,
                bottom_right,
                color="green",
                dash=(10, 10),
                width=5,
            )

            # Now self.annotated_image has the green rectangle drawn around the focus region
            return self.annotated_image

    def get_annotation_df(self):
        """Return a dataframe containing the annotations of the focus region. Must have columns ['TL_x', 'TL_y', 'BR_x', 'BR_y']."""

        if self.wbc_candidate_bboxes is None:
            raise self.FocusRegionNotAnnotatedError

        else:
            return pd.DataFrame(
                self.wbc_candidate_bboxes, columns=["TL_x", "TL_y", "BR_x", "BR_y"]
            )

    def _save_YOLO_df(self, save_dir):
        """Save the YOLO_df as a csv file in save_dir/focus_regions/YOLO_df/self.idx.csv."""

        if self.YOLO_df is None:
            raise self.FocusRegionNotAnnotatedError

        else:
            self.YOLO_df.to_csv(
                os.path.join(save_dir, "focus_regions", "YOLO_df", f"{self.idx}.csv")
            )

    def get_classification(self):
        """Return the classification of the focus region.
        which one of the following:
        - adequate
        - inadequate

        has the highest confidence score.
        """

        if self.adequate_confidence_score is None:
            raise self.FocusRegionNotAnnotatedError

        if self.adequate_confidence_score > region_clf_conf_thres:
            return "adequate"
        else:
            return "inadequate"

    def save_high_mag_image(self, save_dir, annotated=True):
        """Save the high magnification image of the focus region."""

        if self.image is None:
            raise self.FocusRegionNotAnnotatedError

        else:
            if not annotated:
                if self.image is None:
                    raise ValueError(
                        "This FocusRegion object does not possess a high magnification image attribute."
                    )
                self.image.save(
                    os.path.join(
                        save_dir,
                        "focus_regions",
                        "high_mag_unannotated",
                        f"{self.idx}.jpg",
                    )
                )
            else:
                self.get_annotated_image().save(
                    os.path.join(
                        save_dir,
                        "focus_regions",
                        "high_mag_annotated",
                        f"{self.idx}.jpg",
                    )
                )
                self.image.save(
                    os.path.join(
                        save_dir,
                        "focus_regions",
                        "high_mag_unannotated",
                        f"{self.idx}.jpg",
                    )
                )
                self.downsampled_image.save(
                    os.path.join(
                        save_dir,
                        "focus_regions",
                        "low_mag_selected",
                        f"{self.idx}.jpg",
                    )
                )

    class FocusRegionNotAnnotatedError(Exception):
        """Raise when the focus region is not annotated yet."""

        def __init__(self, message="The focus region is not annotated yet."):
            """Initialize a FocusRegionNotAnnotatedError object."""

            super().__init__(message)


def min_resnet_conf(focus_regions):
    """Return the minimum resnet confidence score of the focus regions."""

    minimum = 1

    for focus_region in focus_regions:
        if focus_region.resnet_confidence_score is not None:
            minimum = min(minimum, focus_region.resnet_confidence_score)

    return minimum


def sort_focus_regions(focus_regions):
    """Sort the focus regions by their resnet confidence score largest to smallest."""

    return sorted(
        focus_regions,
        key=lambda focus_region: focus_region.resnet_confidence_score,
        reverse=True,
    )


@ray.remote
def save_focus_region_batch(focus_regions, save_dir):
    """
    Ray task to save a single focus region image.
    """

    for idx, focus_region in enumerate(focus_regions):
        classification = focus_region.get_classification()
        save_path = os.path.join(
            save_dir, "focus_regions", classification, f"{focus_region.idx}.png"
        )

        focus_region.downsampled_image.save(save_path)

    return len(focus_regions)