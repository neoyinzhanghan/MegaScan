import openslide
from MS.resources.BMAassumptions import topview_level
from MS.vision.processing import read_with_timeout
from MS.brain.SpecimenClf import get_specimen_type, calculate_specimen_conf
from MS.PBCounter import PBCounter
from MS.BMACounter import BMACounter


def get_specimen_conf_dict(slide_path):
    """Get the confidence score for each specimen type of the slide. """

    try:
        wsi = openslide.OpenSlide(slide_path)
        top_view = read_with_timeout(
            wsi, (0, 0), topview_level, wsi.level_dimensions[topview_level]
        )

        return calculate_specimen_conf(top_view)
    
    except Exception as e:
        print(e)
        print(
            f"Could not get the confidence scores of {slide_path}, which means it will be classified as Others"
        )
        return {
            "Bone Marrow Aspirate": 0,
            "Peripheral Blood": 0,
            "Manual Peripheral Blood or Inadequate Bone Marrow Aspirate": 0,
            "Others": 2,
        } ### We are going to record the confidence of Others as 2


def classify_specimen_type(slide_path):
    """Get the top view of the wsi and classify it"""

    try:
        wsi = openslide.OpenSlide(slide_path)
        top_view = read_with_timeout(
            wsi, (0, 0), topview_level, wsi.level_dimensions[topview_level]
        )

        specimen_type = get_specimen_type(top_view)

    except Exception as e:
        print(e)
        print(
            f"Could not classify the specimen type of {slide_path}, which means it will be classified as Others"
        )
        specimen_type = "Others"

    if specimen_type == "Bone Marrow Aspirate":
        return "BMA"

    elif specimen_type == "Peripheral Blood":
        return "PB"

    elif specimen_type == "Manual Peripheral Blood or Inadequate Bone Marrow Aspirate":
        return "MPBorIBMA"

    else:
        return "Others"


def heme_analyze(
    slide_path, hoarding=False, continue_on_error=False, do_extract_features=False
):
    """First classify the slide specimen type.
    --If BMA, then use BMACounter to tally differential.
    --If PB, then use PBCounter to tally differential.
    --If MPB or IBMA, then use BMACounter to tally differential.
    --If Others, then do nothing.

    In all situations, return the specimen type.
    """

    # classify the slide specimen type
    specimen_type = classify_specimen_type(slide_path)

    print("Recorded Specimen Type:", specimen_type)

    if specimen_type == "BMA":
        # use BMACounter to tally differential
        bma_counter = BMACounter(
            slide_path,
            hoarding=hoarding,
            continue_on_error=continue_on_error,
            do_extract_features=do_extract_features,
        )
        bma_counter.tally_differential()

    elif specimen_type == "PB":
        # use PBCounter to tally differential
        pb_counter = PBCounter(
            slide_path,
            hoarding=hoarding,
            continue_on_error=continue_on_error,
            do_extract_features=do_extract_features,
        )
        pb_counter.tally_differential()

    elif specimen_type == "MPBorIBMA":
        # use BMACounter to tally differential
        bma_counter = BMACounter(
            slide_path,
            hoarding=hoarding,
            continue_on_error=continue_on_error,
            do_extract_features=do_extract_features,
        )
        bma_counter.tally_differential()

    return specimen_type
