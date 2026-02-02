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
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Cnn import *


class GenderDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filepath"]
        gender = self.df.loc[idx, "gender"]   # 0 = male, 1 = female

        img = Image.open(img_path).convert("RGB")
        img = img.resize(size=(224,224))

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(gender, dtype=torch.long)


def get_data_frame(image_dir):
    data = []
    filenames = os.listdir(image_dir)

    for filename in filenames:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            parts = filename.split('_')

            # UTKFace format requires at least 4 parts
            if len(parts) >= 4:
                try:
                    gender = int(parts[1])
                    if gender not in (0,1):
                        continue
                except:
                    continue

                data.append([
                    os.path.join(image_dir, filename),
                    gender
                ])

    # Return as list of [filepath, gender]
    return data

def save_confusion_matrix(model, dataloader, device):
    all_true = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for imgs, ages in dataloader:
            imgs, ages = imgs.to(device), ages.to(device).long()
            preds = model(imgs)
            pred_classes = preds.argmax(dim=1)
            all_true.extend(ages.cpu().numpy())
            all_pred.extend(pred_classes.cpu().numpy())

    cm = confusion_matrix(all_true, all_pred, labels=list(range(2)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    accuracy = total_correct / total_samples * 100
    plt.text(
    0.5, -0.1, f'Total Accuracy: {accuracy:.2f}%', 
    fontsize=12, ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.title('Confusion Matrix (Validation)')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=100)
    plt.close()

