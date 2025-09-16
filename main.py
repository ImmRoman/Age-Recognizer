
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

image_dir = "dataset"

# List all image files
filenames = os.listdir(image_dir)


# Create a list to hold the parsed data
data = []
for filename in filenames:
    # Ensure it's a valid image file
    if filename.endswith('.jpg'):
        # Split the filename by underscore to get the age
        parts = filename.split('_')
        if len(parts) > 0:
            try:
                age = int(parts[0])
                data.append([os.path.join(image_dir, filename), age])
            except ValueError:
                # Skip files where the age part is not a number
                continue

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=['filepath', 'age'])

def split_db_2to1_df(df: pd.DataFrame, seed: int = 0):
    """
    Splits a DataFrame with 2 columns [data, label] into 2/3 training and 1/3 validation sets.
    
    Args:
        df (pd.DataFrame): First column = data, second column = label.
        seed (int): Random seed for reproducibility.
    
    Returns:
        (DTR, LTR), (DVAL, LVAL)
    """
    n_train = int(len(df) * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]
    
    DTR = df.iloc[idx_train, 0].to_numpy()
    LTR = df.iloc[idx_train, 1].to_numpy()
    DVAL = df.iloc[idx_val, 0].to_numpy()
    LVAL = df.iloc[idx_val, 1].to_numpy()
    
    return (DTR, LTR), (DVAL, LVAL)



class AgeCNN(nn.Module):
    def __init__(self):
        super(AgeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) 

        # ✅ Global pooling → works for any input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)   # (B, 128, 1, 1)
        x = x.view(x.size(0), -1) # (B, 128)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)           # (B, 1)
        return x

# Example transforms
transform = T.Compose([
    T.Resize((128,128)),   # you can change to (224,224)
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
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

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, torch.tensor(age, dtype=torch.float32)


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
for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, ages in train_loader:
        imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, ages)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, ages in val_loader:
            imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, ages)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")

    # Compute accuracy (percentage of predictions within ±5 years of true age)
    correct = 0
    total = 0
    # Compute bucketed accuracy (predicted and true ages in same 10-year bucket)
    bucket_correct = 0
    bucket_total = 0
    with torch.no_grad():
        for imgs, ages in val_loader:
            imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
            preds = model(imgs)
            pred_buckets = (preds // 10).int()
            true_buckets = (ages // 10).int()
            bucket_correct += (pred_buckets == true_buckets).sum().item()
            bucket_total += ages.size(0)
    bucket_accuracy = bucket_correct / bucket_total if bucket_total > 0 else 0
    print(f"Validation bucket accuracy (same 10-year bucket): {bucket_accuracy*100:.2f}%")