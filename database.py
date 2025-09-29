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

        img = Image.open(img_path).convert("RGB")
        #Fatto per i dataloader di immagini non nel dataset
        img = img.resize(size=(200,200))
        #converte da BGR a RGB coi db salvati da opencv 
        # r, g, b = img.split()
        # img_rgb = Image.merge("RGB", (b, g, r))

        # img = Image.open(img_path).convert("L") #grayscale
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(age, dtype=torch.long)


def get_data_frame(image_dir):
    data = []
    contatore = 0
    filenames = os.listdir(image_dir)
    for filename in filenames:
        contatore += 1
        # if(contatore % 5 != 0):
        #     continue
        #Ensure it's a valid image file
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

def plot_confusion_matrix(model, dataloader, device):
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

    cm = confusion_matrix(all_true, all_pred,labels=list(range(8)))
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
    plt.show()

