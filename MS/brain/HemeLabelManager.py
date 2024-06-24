#####################################################################################################
# Imports ###########################################################################################
#####################################################################################################

# Outside imports ##################################################################################
import torch
import torch.nn as nn
import ray
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms
from collections import OrderedDict

# Within package imports ###########################################################################
from MS.resources.BMAassumptions import *


# class Myresnext50(nn.Module):
#     def __init__(self, my_pretrained_model, num_classes=23):
#         super(Myresnext50, self).__init__()
#         self.pretrained = my_pretrained_model
#         self.my_new_layers = nn.Sequential(
#             nn.Linear(1000, 100), nn.ReLU(), nn.Linear(100, num_classes)
#         )
#         self.num_classes = num_classes

#     def forward(self, x):
#         x = self.pretrained(x)
#         x = self.my_new_layers(x)

#         pred = torch.sigmoid(x.reshape(x.shape[0], 1, self.num_classes))
#         return pred


class Myresnext50(nn.Module):
    def __init__(self, my_pretrained_model, num_classes=23):
        super(Myresnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(
            nn.Linear(1000, 100), nn.ReLU(), nn.Linear(100, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x, return_features=False):
        features = self.pretrained(x)
        x = self.my_new_layers(features)
        pred = torch.sigmoid(x.reshape(x.shape[0], 1, self.num_classes))

        if return_features:
            return pred, features
        else:
            return pred


def model_create(num_classes=23, path="not_existed_path"):
    resnext50_pretrained = torch.hub.load("pytorch/vision:v0.10.0", "resnext50_32x4d")
    My_model = Myresnext50(
        my_pretrained_model=resnext50_pretrained, num_classes=num_classes
    )

    checkpoint_PATH = path
    checkpoint = torch.load(checkpoint_PATH)  # , map_location=torch.device("cpu"))

    checkpoint = remove_data_parallel(checkpoint["model_state_dict"])

    My_model.load_state_dict(checkpoint, strict=True)

    My_model.eval()
    My_model.to("cuda")

    return My_model


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`

        new_state_dict[name] = v

    return new_state_dict


def predict_on_cpu(image, model):
    # Define the transformations

    # make sure the image is RGB if it is not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_transforms = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize([0.5594, 0.4984, 0.6937], [0.2701, 0.2835, 0.2176]),
        ]
    )

    # Apply transformations to the input image and create a batch
    image = image_transforms(image).float().unsqueeze(0)

    # Set the model to evaluation mode and make predictions
    model.to("cpu")
    model.eval()

    # Move the image to the CPU if available
    device = torch.device("cpu")

    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    # Process the output as in the original code snippet
    output = torch.flatten(output, start_dim=1).detach().cpu().numpy()
    prediction = tuple(output[0])

    # Return the prediction
    return prediction


def predict_batch(pil_images, model):
    # Define the transformations
    image_transforms = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize([0.5594, 0.4984, 0.6937], [0.2701, 0.2835, 0.2176]),
        ]
    )

    # Apply transformations to each image and create a batch
    batch = torch.stack([image_transforms(image).float() for image in pil_images])

    # Move the batch to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    # Process each output as in the original code snippet
    predictions = []
    for output in outputs:
        output = torch.flatten(output, start_dim=1).detach().cpu().numpy()
        predictions.append(tuple(output[0]))

    # Return a list of predictions in the same order as the input images
    return predictions


def get_features_batch(pil_images, model):
    # Define the transformations
    image_transforms = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5594, 0.4984, 0.6937], [0.2701, 0.2835, 0.2176]),
        ]
    )

    # Apply transformations to each image and create a batch
    batch = torch.stack([image_transforms(image).float() for image in pil_images])

    # Move the batch to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        _, outputs = model(batch, return_features=True)

    # Process each output as in the original code snippet
    features = []
    for output in outputs:
        # print the shape of the output
        # print(output.shape)
        output = output.detach().cpu().numpy()
        features.append(output)

    # Return a list of predictions in the same order as the input images
    return features


# @ray.remote(num_gpus=num_gpus_per_manager, num_cpus=num_cpus_per_manager)
@ray.remote(num_gpus=1)
class HemeLabelManager:
    """A class representing a HemeLabel Manager that manages the classification of a WSI.

    === Class Attributes ===
    - model : the HemeLabel model
    - ckpt_path : the path to the checkpoint of the HemeLabel model
    - num_classes : the number of classes of the HemeLabel model
    """

    def __init__(self, ckpt_path, num_classes=23) -> None:
        """Initialize the HemeLabelManager object."""

        self.model = model_create(num_classes=num_classes, path=ckpt_path)
        self.ckpt_path = ckpt_path
        self.num_classes = num_classes

    def async_label_wbc_candidate(self, wbc_candidate):
        """Label a WBC candidate."""

        image_transforms = transforms.Compose(
            [
                transforms.Resize(96),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     [0.5594, 0.4984, 0.6937], [0.2701, 0.2835, 0.2176]
                # ),
            ]
        )

        if do_zero_pad:
            image = wbc_candidate.padded_YOLO_bbox_image
        else:
            image = wbc_candidate.snap_shot

        self.model.eval()
        # self.model.to("cpu")
        # self.model.to("cuda") # commented for debugging # TODO we need GPU implementation

        image = image_transforms(image).float().unsqueeze(0)

        # move the image to a cuda tensor
        image = image.to(
            "cuda"
        )  # commented for debugging # TODO we need GPU implementation

        ### BELOW MAY BECOME DEPRECATED ###

        # image = np.array(image)
        # image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)

        # image = np.einsum('ijk->kij', image)

        # image = image / 255.0
        # # image = np.transpose(image, (2, 0, 1))
        # image = torch.from_numpy(image).float().unsqueeze(0)

        # image = image.to("cuda") # commented for debugging # TODO we need GPU implementation
        output = self.model(image)
        output = torch.flatten(output, start_dim=1).detach().cpu().numpy()

        # make a clone of the output vector, use tuple to avoid deprecation and aliasing errors down the road
        wbc_candidate.softmax_vector = tuple(output[0])

        return wbc_candidate

    def async_label_wbc_candidate_batch(self, wbc_candidates):
        processed_wbc_candidates = []

        if not do_zero_pad:
            pil_images = [wbc_candidate.snap_shot for wbc_candidate in wbc_candidates]
        else:
            pil_images = [
                wbc_candidate.padded_YOLO_bbox_image for wbc_candidate in wbc_candidates
            ]

        results = predict_batch(pil_images, self.model)

        for i, wbc_candidate in enumerate(wbc_candidates):
            wbc_candidate.softmax_vector = results[i]
            processed_wbc_candidates.append(wbc_candidate)

        return processed_wbc_candidates

    def async_save_wbc_image_feature_batch(self, image_paths):
        """For each image, save the image;s feature vector to save_dir."""

        # first read in the batch of images using PIL
        if not do_zero_pad:
            pil_images = [Image.open(image_path) for image_path in image_paths]
        else:
            pil_images = [
                Image.open(image_path).convert("RGB") for image_path in image_paths
            ]

        features = get_features_batch(pil_images, self.model)

        # for each image, save the feature vector
        for i, image_path in enumerate(image_paths):
            # save the feature vector as a torch tensor
            os.makedirs(
                os.path.join(os.path.dirname(image_path), "features"), exist_ok=True
            )

            save_path = os.path.join(
                os.path.dirname(image_path),
                "features",
                os.path.basename(image_path).replace(".jpg", ".pt"),
            )

            torch.save(
                features[i],
                save_path,
            )

        return image_paths
