####################################################################################################
# Imports ##########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openslide
from pathlib import Path
from PIL import Image

# Within package imports ###########################################################################
from MS.vision.masking import get_white_mask, get_obstructor_mask, get_top_view_mask
from MS.resources.PBassumptions import *
from MS.vision.processing import read_with_timeout


def extract_top_view(wsi_path, save_dir=None):
    # you can get the stem by removing the last 5 characters from the file name (".ndpi")
    stem = Path(wsi_path).stem[:-5]

    print("Extracting top view")
    # open the wsi in tmp_dir and extract the top view
    wsi = openslide.OpenSlide(wsi_path)
    toplevel = wsi.level_count - 1
    topview = read_with_timeout(
        wsi=wsi,
        location=(0, 0),
        level=toplevel,
        dimensions=wsi.level_dimensions[toplevel],
    )

    # make sure to convert topview tp a PIL image in RGB mode
    if topview.mode != "RGB":
        topview = topview.convert("RGB")

    if save_dir is not None:
        topview.save(os.path.join(save_dir, stem + ".jpg"))
    wsi.close()

    return topview


class TopView:

    """A TopView class object representing all the information needed at the top view of the WSI.

    === Class Attributes ===
    - image : the image of the top view
    - obstructor_mask : the obstructor mask of the top view
    - white_mask : the white mask of the top view
    - top_view_mask : the top view mask of the top view for downstream focus region selection
    - width : the width of the top view
    - height : the height of the top view
    - downsampling_rate : the downsampling rate of the top view
    - level : the level of the top view in the WSI

    - is_bma : whether the top view is a bone marrow aspirate top view
    - verbose : whether to print out the progress of the top view
    """

    def __init__(self, image, downsampling_rate, level, verbose=False, is_bma=False):
        """Initialize a TopView object.
        Image is a PIL image. Check the type of image. If not PIL image, raise ValueError.
        """
        self.verbose = verbose
        self.is_bma = is_bma

        if self.verbose:
            print("Checking the type of image...")
        # check the type of image
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL image.")

        self.image = image

        if self.verbose:
            print("Printing various masks of the top view...")

        if not is_bma:
            try:
                self.obstructor_mask = get_obstructor_mask(image)
                self.white_mask = get_white_mask(image)
                self.top_view_mask = get_top_view_mask(
                    image,
                    obstructor_mask=self.obstructor_mask,
                    white_mask=self.white_mask,
                )
            except Exception as e:
                print(e)
                # the obstructor mask should be 1 everywhere
                # the white mask should be 0 everywhere
                # the top view mask should be 1 everywhere
                self.obstructor_mask = (
                    np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
                )
                self.white_mask = np.zeros(
                    (image.size[1], image.size[0]), dtype=np.uint8
                )
                self.top_view_mask = (
                    np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
                )

            # if the proportion of white pixels in the top view mass is less than min_top_view_mask_prop, then change the top view mask to be 1 everywhere
            if (
                self.top_view_mask.sum() / (self.top_view_mask.size * 255)
                < min_top_view_mask_prop
            ):
                self.obstructor_mask = (
                    np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
                )
                self.white_mask = np.zeros(
                    (image.size[1], image.size[0]), dtype=np.uint8
                )
                self.top_view_mask = (
                    np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
                )

            self.width = image.size[0]
            self.height = image.size[1]
            self.downsampling_rate = downsampling_rate

        else:
            self.obstructor_mask = (
                np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
            )
            self.white_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            self.top_view_mask = (
                np.ones((image.size[1], image.size[0]), dtype=np.uint8) * 255
            )

            self.width = image.size[0]
            self.height = image.size[1]
            self.downsampling_rate = downsampling_rate

    def is_peripheral_blood(self):
        """Return True iff the top view is a peripheral blood top view."""
        return True

    def save_images(self, save_dir):
        """Save the top view image and the top_view_mask in save_dir."""

        # Save the original image
        self.image.save(os.path.join(save_dir, "top_view_image.png"))

        # the top_view_mask is a numpy array in grayscale, so we need to convert it to a PIL image
        mask_image = Image.fromarray(self.top_view_mask)

        # If the mask is not in 'L' mode, convert it to 'L'
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")

        # Create an RGBA version of the top view image
        overlay_image = Image.new("RGBA", self.image.size)
        overlay_image.paste(self.image.convert("RGBA"))  # Convert to RGBA and paste

        # Create the green mask where the mask is not 0
        green_mask = np.zeros((*self.top_view_mask.shape, 4), dtype=np.uint8)
        green_mask[self.top_view_mask != 0] = [0, 255, 0, 100]  # green color with alpha

        # Convert the green mask to a PIL Image
        green_pil_mask = Image.fromarray(green_mask, mode="RGBA")

        # Overlay the green mask onto the top view image
        overlayed_image = Image.alpha_composite(overlay_image, green_pil_mask)

        # Save the overlayed image
        overlayed_image.save(os.path.join(save_dir, "top_view_overlayed_image.png"))

    def _crop_using_connected_components(
        self,
        blue_intensity,
        outliers,
        location,
        top_to_search_zoom_ratio,
        image,
        search_to_0_zoom_ratio,
        focus_regions_size,
        padding_x,
        padding_y,
        verbose=False,
    ):
        # find the connected components of the outliers
        # Create a mask of the outliers
        mask = np.zeros_like(blue_intensity, dtype=np.uint8)
        mask[outliers] = 255

        # Find the connected components of the outliers
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        if verbose:
            # Print how many connected components were found
            print(f"Found {num_labels} connected components")

        # remove the centroid of the whole image
        centroids = centroids[1:]

        filtered_centroids = []
        removed_centroids = []

        # check if the centroid coordinate, scaled down by zoom_ratio, is in the top_view_mask, if not remove it
        for centroid in centroids:
            scaled_centroid_0 = (centroid[0] + location[0]) / top_to_search_zoom_ratio
            scaled_centroid_1 = (centroid[1] + location[1]) / top_to_search_zoom_ratio
            mask_value = self.top_view_mask[
                int(scaled_centroid_1), int(scaled_centroid_0)
            ]
            # print(f"Scaled Centroid: ({scaled_centroid_0}, {scaled_centroid_1}), Mask Value: {mask_value}")
            if mask_value > 0:
                filtered_centroids.append(centroid)
            else:
                removed_centroids.append(centroid)

        if verbose:
            print(f"Filtered out {len(centroids) - len(filtered_centroids)} centroids")

        # Create a list of the centroid pixel of each connected component
        for i in range(len(filtered_centroids)):
            filtered_centroids[i][0] += location[0]
            filtered_centroids[i][1] += location[1]

        for i in range(len(removed_centroids)):
            removed_centroids[i][0] += location[0]
            removed_centroids[i][1] += location[1]

        # plot the filtered_centroids on the original image with centroids in green and alpla=0.2
        # plot also the removed_centroids in red and alpha=0.2
        if verbose:
            plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.scatter(
                [centroid[0] - location[0] for centroid in filtered_centroids],
                [centroid[1] - location[1] for centroid in filtered_centroids],
                s=75,
                c="green",
                alpha=0.2,
            )
            plt.scatter(
                [centroid[0] - location[0] for centroid in removed_centroids],
                [centroid[1] - location[1] for centroid in removed_centroids],
                s=75,
                c="red",
                alpha=0.2,
            )
            plt.show()

        # print the filtered centroids
        if verbose:
            print(f"Filtered centroids: {filtered_centroids}")
        # print the removed centroids
        if verbose:
            print(f"Removed centroids: {removed_centroids}")

        # use search_to_0_zoom_ratio to scale the focus_region_size
        scaled_focus_region_size = focus_regions_size / search_to_0_zoom_ratio

        # get the corner coordinates of the region_crop using crop_location
        TL_x, TL_y, BR_x, BR_y = location[0], location[1], location[2], location[3]

        # imagine crop_location is actually living inside a larger image, that image is being cut into blocks of size scaled_focus_region_size
        # get a list of (TL_x, TL_y) coordinates of blocks whose centroids are are inside the coordinates of region_crop
        # the coordinates of the blocks are in the coordinate system of the larger image

        # get the coordinates of the blocks centroids
        block_coords = []

        # get the number of blocks in the x and y directions

        where_blocking_index_starts_x = (TL_x - padding_x) // scaled_focus_region_size
        where_blocking_starts_x = (
            where_blocking_index_starts_x * scaled_focus_region_size + padding_x
        )

        where_blocking_index_starts_y = (TL_y - padding_y) // scaled_focus_region_size
        where_blocking_starts_y = (
            where_blocking_index_starts_y * scaled_focus_region_size + padding_y
        )

        num_blocks_x = int(
            (BR_x - where_blocking_starts_x) // scaled_focus_region_size + 1
        )
        num_blocks_y = int(
            (BR_y - where_blocking_starts_y) // scaled_focus_region_size + 1
        )

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                block_TL_x = (
                    where_blocking_index_starts_x + i
                ) * scaled_focus_region_size + padding_x
                block_TL_y = (
                    where_blocking_index_starts_y + j
                ) * scaled_focus_region_size + padding_y

                block_BR_x = block_TL_x + scaled_focus_region_size
                block_BR_y = block_TL_y + scaled_focus_region_size

                # calculate centroid of the block
                block_centroid_x = (block_TL_x + block_BR_x) / 2
                block_centroid_y = (block_TL_y + block_BR_y) / 2

                # check if the centroid is inside the region_crop
                if (
                    TL_x <= block_centroid_x <= BR_x
                    and TL_y <= block_centroid_y <= BR_y
                ):
                    # check if any of the filtered_centroids is inside the block
                    contain_good_stuff = False
                    for centroid in filtered_centroids:
                        if (
                            block_TL_x <= centroid[0] <= block_BR_x
                            and block_TL_y <= centroid[1] <= block_BR_y
                        ):
                            contain_good_stuff = True
                            break

                    if contain_good_stuff:
                        block_coords.append(
                            (
                                int(block_TL_x * search_to_0_zoom_ratio),
                                int(block_TL_y * search_to_0_zoom_ratio),
                                int(block_BR_x * search_to_0_zoom_ratio),
                                int(block_BR_y * search_to_0_zoom_ratio),
                            )
                        )

        return block_coords

    def find_focus_regions(
        self,
        crop,
        location,
        focus_regions_size,
        padding_x,
        padding_y,
        num_sds=foci_sds,
        top_to_search_zoom_ratio=16,
        search_to_0_zoom_ratio=8,
        verbose=False,
    ):
        """Find the focus regions related to the crop. Return a list of focus region level 0 coordinates in the format of (TL_x, TL_y, BR_x, BR_y)."""

        # make sure to convert from PIL to a numpy array of BGR
        crop_clone = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

        # Make sure to convert to float32
        image = crop_clone.astype(np.float32)

        # Get the sum of all three channels and add small constant to avoid division by zero
        # add a small constant to prevent division by zero
        total = image.sum(axis=2) + 1e-8

        # Get the relative blue pixel intensity, which is the blue channel divided by the sum of all three channels
        blue_intensity = image[:, :, 0] / total

        # Make sure to convert to int8
        blue_intensity = (blue_intensity * 255).astype(np.uint8)

        # apply a gaussian blurring filter to the image
        blue_intensity = cv2.GaussianBlur(blue_intensity, (15, 15), 0)

        if verbose:
            # Plot the original image and the blue intensity side by side
            plt.subplot(1, 2, 1)
            # cv2 reads image in BGR format
            plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.subplot(1, 2, 2)
            plt.imshow(blue_intensity, cmap="gray")
            plt.show()

        # identify the outliers in the blue intensity image
        # Get the mean and standard deviation of the blue intensity
        mean, std = cv2.meanStdDev(blue_intensity)

        # Get the lower and upper bounds of the blue intensity
        upper_bound = mean + num_sds * std

        # Get the right-side outliers
        outliers = np.where(blue_intensity > upper_bound)

        if verbose:
            # Plot the original image, the blue intensity, and the outliers side by side
            plt.subplot(1, 3, 1)
            # cv2 reads image in BGR format
            plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.subplot(1, 3, 2)
            plt.imshow(blue_intensity, cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(blue_intensity, cmap="gray", alpha=0)
            plt.scatter(outliers[1], outliers[0], s=1, c="blue")
            plt.show()

        #############################################################################
        # This is the piece of code that we are to potentially repeat with while loop
        #############################################################################

        block_coords = self._crop_using_connected_components(
            blue_intensity,
            outliers,
            location,
            top_to_search_zoom_ratio,
            image,
            search_to_0_zoom_ratio,
            focus_regions_size,
            padding_x,
            padding_y,
            verbose=verbose,
        )

        return block_coords


class SpecimenError(ValueError):
    """Exception raised when the specimen is not the correct type for the operation."""

    pass


class RelativeBlueSignalTooWeakError(ValueError):
    """Exception raised when the blue signal is too weak."""

    def __init__(self, message):
        """Initialize a BlueSignalTooWeakError object."""

        super().__init__(message)

    def __str__(self):
        """Return the error message."""

        return self.args[0]
