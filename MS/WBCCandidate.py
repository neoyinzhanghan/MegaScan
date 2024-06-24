####################################################################################################
# Imports ##########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import os
import pandas as pd
import numpy as np
import torch

# Within package imports ###########################################################################
from MS.vision.image_quality import VoL
from MS.resources.PBassumptions import *


class WBCCandidate:
    """A class representing a WBC candidate.

    === Class Attributes ===

    - snap_shot_bbox : The bounding box of the snap shot of the candidate, in the level_0 view of the PBCounter object containing this candidate.
    - YOLO_bbox : The bounding box of the candidate, in the level_0 view of the PBCounter object containing this candidate, in the format of (TL_x, TL_y, BR_x, BR_y)
    - YOLO_bbox_image : The image of the candidate, in the level_0 view of the PBCounter object containing this candidate.

    - snap_shot : The snap shot of the candidate, in the search_view of the PBCounter object containing this candidate.
    - padded_YOLO_bbox_image : The padded bounding box of the candidate, cropped and zero padded to have square dimension of snap_shot_size

    - confidence : The confidence of the candidate, in the search_view of the PBCounter object containing this candidate.
    - VoL : The variance of laplacian of the snap shot of the candidate, in the level_0 view of the PBCounter object containing this candidate.

    - softmax_vector : the director softmax vector output of the HemeLabel model, should be a vector of length 23
    - name : the name of the candidate, should be the cell_id followed by top 4 classes separated by dashes, with extension .jpg for example: 1-ER4-ER5-ER2-ER1.jpg
    - cell_id : the cell_id of the candidate, should be an integer
    - cell_df_row: a pandas dataframe row of the cell_df of the PBCounter object containing this candidate
        - the dataframe should have the following columns: [cell_id, name, coords, confidence, VoL, cellnames[0], ..., cellnames[num_classes - 1]

    - idx : the cell idx, None when initialized and will be assigned eventually
    - features : a dictionary of features of the candidate the key is a specific feature extraction architecture and the value is the feature vector
    - augmented_features : a dictionary of augmented features of the candidate the key is a specific feature extraction architecture and the value is a list of triples (augmentation_pipeline, augmented_image, augmented_feature_vector)
    """

    def __init__(
        self,
        snap_shot,
        YOLO_bbox_image,
        padded_YOLO_bbox_image,
        snap_shot_bbox,
        YOLO_bbox,
        confidence,
        local_idx,
        focus_region_idx,
    ):
        """Initialize a WBCCandidate object."""

        self.local_idx = local_idx
        self.focus_region_idx = focus_region_idx

        self.snap_shot_bbox = snap_shot_bbox
        self.YOLO_bbox = YOLO_bbox
        self.YOLO_bbox_image = YOLO_bbox_image

        self.snap_shot = snap_shot
        self.padded_YOLO_bbox_image = padded_YOLO_bbox_image
        self.VoL = VoL(snap_shot)

        self.confidence = confidence
        self.softmax_vector = None
        self.name = None
        self.cell_df_row = None

        self.features = {}
        self.augmented_features = {}

    def compute_cell_info(self):
        """Return a pandas dataframe row of the cell_df of the PBCounter object containing this candidate."""

        if self.softmax_vector is None:
            raise CellNotClassifiedError("The softmax vector is not computed yet.")

        elif self.cell_df_row is not None:
            return self.cell_df_row

        else:
            sofmax_vector_np = np.array(self.softmax_vector)

            # the cell classes should come first in the name, and then followed by the focus region idx and the local idx
            self.name = (
                "-".join([cellnames[i] for i in sofmax_vector_np.argsort()[-4:][::-1]])
                + "_"
                + str(self.focus_region_idx)
                + "-"
                + str(self.local_idx)
                + ".jpg"
            )

            cell_df_row = [
                self.focus_region_idx,
                self.local_idx,
                self.name,
                self.YOLO_bbox[0],
                self.YOLO_bbox[1],
                self.YOLO_bbox[2],
                self.YOLO_bbox[3],
                self.confidence,
                self.VoL,
            ] + list(self.softmax_vector)

            # convert the list to a pandas dataframe row
            self.cell_df_row = pd.DataFrame(
                [cell_df_row],
                columns=[
                    "focus_region_idx",
                    "local_idx",
                    "name",
                    "TL_x",
                    "TL_y",
                    "BR_x",
                    "BR_y",
                    "confidence",
                    "VoL",
                ]
                + [cellnames[i] for i in range(num_classes)],
            )

            return self.cell_df_row

    def _save_YOLO_bbox_image(self, save_dir, subfolder="blurry"):
        """Save the YOLO_bbox_image to the save_dir.
        The file name should just be cell id since the cell may not be classified yet.

        Precondition: the cell is not classified yet.
        """

        self.YOLO_bbox_image.save(
            os.path.join(
                save_dir,
                "cells",
                subfolder,
                "VoL"
                + str(round(self.VoL))
                + str(self.focus_region_idx)
                + "-"
                + str(self.local_idx)
                + ".jpg",
            )
        )

    def _save_cell_image(self, save_dir):
        """Save the snap_shot to the save_dir/cells/class where class is the class of the cell.

        Precondition: the cell is classified.
        """

        if self.softmax_vector is None:
            raise CellNotClassifiedError("The softmax vector is not computed yet.")

        elif self.name is None:
            raise CellNotClassifiedError("The name is not computed yet.")

        else:
            os.makedirs(
                os.path.join(
                    save_dir, "cells", cellnames[np.argmax(self.softmax_vector)]
                ),
                exist_ok=True,
            )
            self.snap_shot.save(
                os.path.join(
                    save_dir,
                    "cells",
                    cellnames[np.argmax(self.softmax_vector)],
                    self.name,
                )
            )

    def _save_cell_feature(self, save_dir, arch):
        """Save the feature vector to the save_dir/arch/class directory.

        Precondition: the cell is classified and the feature vector is computed.
        """

        if self.softmax_vector is None:
            raise CellNotClassifiedError("The softmax vector is not computed yet.")

        elif self.name is None:
            raise CellNotClassifiedError("The name is not computed yet.")

        else:
            os.makedirs(os.path.join(save_dir, arch), exist_ok=True)
            os.makedirs(
                os.path.join(save_dir, arch, cellnames[np.argmax(self.softmax_vector)]),
                exist_ok=True,
            )
            torch.save(
                self.features[arch],
                os.path.join(
                    save_dir,
                    arch,
                    cellnames[np.argmax(self.softmax_vector)],
                    self.name.replace(".jpg", ".pt"),
                ),
            )


class CellNotClassifiedError(ValueError):
    """An error raised when the cell is not classified."""

    def __init__(self, message):
        """Initialize a CellNotClassifiedError object."""

        super().__init__(message)
