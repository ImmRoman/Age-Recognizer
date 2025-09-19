
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
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64,128,3)
        self.fc1 = nn.Linear(25088,256)
        self.fc2 = nn.Linear(256,8)
        self.fc3 = nn.Linear(8,1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)



    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
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




