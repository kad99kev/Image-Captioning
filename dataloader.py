import torch

import pandas as pd
import numpy as np
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.path + self.df.image[idx]))

        if self.transform:
            img = self.transform(img)

        return img