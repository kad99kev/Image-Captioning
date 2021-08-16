import torch
import torchvision

import pandas as pd
import numpy as np
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    """
    An Image Dataset class for a vision model.

    Parameters:
        img_list (list): A list of images.
        path (str): Path of the images.
        transforms (torchvision.transforms.Compose): Transform an image using PyTorch.
    """

    def __init__(
        self, path: str, img_list: list, transform: torchvision.transforms.Compose
    ):
        self.path = path
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(self.path + img_path)

        if self.transform:
            img = self.transform(img)

        return img, img_path


class FeatureCaptionDataset(torch.utils.data.Dataset):
    """
    A Feature Caption Dataset class for a language model.

    Parameters:
        path (str): Path of the features.
        df (pd.DataFrame): Transform an image using PyTorch.
    """

    def __init__(self, path: str, df: pd.DataFrame):
        self.path = path
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature_path = self.path + self.df.image[idx]
        caption = self.df.caption[idx]

        feature = np.load(feature_path + ".npy")

        return feature, caption