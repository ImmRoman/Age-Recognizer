
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
from database import *

# Create a Pandas DataFrame
df = pd.DataFrame(get_data_frame(), columns=['filepath', 'age'])

transform = T.Compose([
    T.Resize((128,128)),   
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

# Dataset
dataset = AgeDataset(df, transform=transform)

# Train/val split
n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeCNN().to(device)
criterion = nn.L1Loss()   # MAE is better for age
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    bucket_correct_train = 0
    bucket_total_train = 0
    for imgs, ages in train_loader:
        imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, ages)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        bucket_correct_train += (preds.int() == ages.int()).sum().item()
        bucket_total_train += ages.size(0)
    bucket_accuracy_train = bucket_correct_train / bucket_total_train
    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        bucket_correct = 0
        bucket_total = 0
        for imgs, ages in val_loader:
            imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, ages)
            val_loss += loss.item() * imgs.size(0)
            bucket_correct += (preds.int() == ages.int()).sum().item()
            bucket_total += ages.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
    bucket_accuracy = bucket_correct / bucket_total if bucket_total > 0 else 0
    #Compute accuracies
    print(f" Testing bucket accuracy (same age bucket): {bucket_accuracy_train*100:.2f}%")
    print(f" Validation bucket accuracy (same age bucket): {bucket_accuracy*100:.2f}%")
    # Save model at the end of each epoch
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/age_cnn_epoch_{epoch+1}.pth")
