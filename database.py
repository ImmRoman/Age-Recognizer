import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset

from Cnn import *

image_dir = "dataset"

# List all image files
filenames = os.listdir(image_dir)

class AgeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        dataframe: pd.DataFrame with columns [filepath, age]
        transform: torchvision transforms to apply
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filepath"]
        age = self.df.loc[idx, "age"]

        #img = Image.open(img_path).convert("RGB")
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(age, dtype=torch.float32)


def get_data_frame():
    data = []
    for filename in filenames:
        # Ensure it's a valid image file
        if filename.endswith('.jpg'):
            # Split the filename by underscore to get the age
            parts = filename.split('_')
            if len(parts) > 0:
                try:
                    age = int(parts[0])
                    data.append([os.path.join(image_dir, filename), get_age_bucket(age)])
                except ValueError:
                    # Skip files where the age part is not a number
                    continue
    return data



