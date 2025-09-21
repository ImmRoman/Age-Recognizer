
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


class AgeCNN(nn.Module):
    def __init__(self):
        super(AgeCNN, self).__init__()
        # Input: grayscale (1 channel), 200x200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.AvgPool2d(2, 2)   # same as AveragePooling2D

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # like GlobalAveragePooling2D

        # Dense layers
        self.fc1 = nn.Linear(256, 132)
        self.fc2 = nn.Linear(132, 9)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.global_pool(x)        
        x = torch.flatten(x, 1)       

        x = F.relu(self.fc1(x))
        x = self.fc2(x)                
        return x


def get_age_bucket(age):
    match age:
        case a if 0<=a<=3:
            return 0
        case a if 4<=a<=7:
            return 1
        case a if 8<=a<=14:
            return 2
        case a if 15<=a<=21:
            return 3
        case a if 15<=a<=21:
            return 4
        case a if 22<=a<=37:
            return 5
        case a if 38<=a<=47:
            return 6
        case a if 48<=a<=59:
            return 7
        case a if a>=60:
            return 8




