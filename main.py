
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
rotations_iterator = [-40,-20,0,20,40,360,-40,-20,0,20,40]
transform = T.Compose([
    T.Resize((200,200)),   
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

# Dataset
dataset = AgeDataset(df, transform=transform)

# Train/val split
n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size = 512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 512)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeCNN().to(device)
# UTKFace approximate bucket distribution (example, adjust if you have exact counts):
# [0-3]: 6000, [4-7]: 5000, [8-14]: 7000, [15-21]: 8000, [22-37]: 12000, [38-47]: 6000, [48-59]: 4000, [60+]: 3000
bucket_counts = torch.tensor([6000, 5000, 7000, 8000, 12000, 6000, 4000, 3000], dtype=torch.float)
bucket_weights = bucket_counts.sum() / (len(bucket_counts) * bucket_counts)
bucket_weights = bucket_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=bucket_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("="*29)
print("====   INIZIO TRAINING   ====")
print("="*29)
# Training loop
for epoch in range(60):
    model.train()
    total_loss = 0
    bucket_correct_train = 0
    bucket_total_train = 0
    for imgs, ages in train_loader:
        imgs, ages = imgs.to(device), ages.to(device).long()
        #Ogni immagine viene mandata nella rete con 5 rotazioni di entrambre le versioni specchiate verticalmente
        for rotation in rotations_iterator :     
            if (rotation == 360):
                imgs = T.functional.vflip(imgs)
            r_img = T.functional.rotate(imgs, angle=rotation)
            preds = model(r_img)

            loss = criterion(preds, ages)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

            pred_classes = preds.argmax(dim=1)   
            bucket_correct_train += (pred_classes == ages).sum().item()
            bucket_total_train += ages.size(0)

    bucket_accuracy_train = bucket_correct_train / bucket_total_train
    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        bucket_correct_validation = 0
        bucket_total_validation = 0
        for imgs, ages in val_loader:
            imgs, ages = imgs.to(device), ages.to(device).long()

            preds = model(imgs)
            loss = criterion(preds, ages)
            val_loss += loss.item() * imgs.size(0)

            pred_classes = preds.argmax(dim=1)   
            bucket_correct_validation += (pred_classes == ages).sum().item()
            bucket_total_validation += ages.size(0)

    val_loss /= len(val_loader.dataset)

   
    
    print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
    bucket_accuracy_validation = bucket_correct_validation / bucket_total_validation if bucket_total_validation > 0 else 0
    #Compute accuracies
    print(f" Testing bucket accuracy (same age bucket): {bucket_accuracy_train*100:.2f}%")
    print(f" Validation bucket accuracy (same age bucket): {bucket_accuracy_validation*100:.2f}%")

    if (epoch%10 == 0):
        # Save model at the end of each epoch
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/age_cnn_epoch_{epoch+1}.pth")
        plot_confusion_matrix(model, val_loader, device)


