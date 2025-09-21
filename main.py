
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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create a Pandas DataFrame
df = pd.DataFrame(get_data_frame(), columns=['filepath', 'age'])

transform = T.Compose([
    T.Resize((256,256)),   
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    bucket_correct_train = 0
    bucket_total_train = 0
    for imgs, ages in train_loader:
        imgs, ages = imgs.to(device), ages.to(device).long()

        optimizer.zero_grad()
        preds = model(imgs)
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

    plot_confusion_matrix(model, val_loader, device)
    
    print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
    bucket_accuracy_validation = bucket_correct_validation / bucket_total_validation if bucket_total_validation > 0 else 0
    #Compute accuracies
    print(f" Testing bucket accuracy (same age bucket): {bucket_accuracy_train*100:.2f}%")
    print(f" Validation bucket accuracy (same age bucket): {bucket_accuracy_validation*100:.2f}%")
    # Save model at the end of each epoch
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/age_cnn_epoch_{epoch+1}.pth")



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
        cm = confusion_matrix(all_true, all_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Validation)')
        plt.show()