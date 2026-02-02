import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch.optim as optim
import torchvision.models as models
from collections import defaultdict

from Cnn import *
from database import *
DATABASE_PATH = "cropped_output" 
EPOCHS = 300 
top_accuracy = 0

# --- Main Execution ---

# Setup MobileNet
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
# Adjust classifier for 8 classes
_in_features = None
for m in mobilenet_v3_large.classifier.modules():
    if isinstance(m, nn.Linear):
        _in_features = m.in_features
        break
if _in_features is None: _in_features = 1024

mobilenet_v3_large.classifier = nn.Sequential(
    nn.Linear(512, 8)
)

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# Load Data
print(f"Looking for data in: {os.path.abspath(DATABASE_PATH)}")
raw_data = get_data_frame(DATABASE_PATH)
if len(raw_data) == 0:
    print("ERROR: No images found. Check your zip extraction path.")
else:
    df = pd.DataFrame(raw_data, columns=['filepath', 'age'])
    
    # Transforms
    transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485 , 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = AgeDataset(df, transform=transform)

    # Split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=4)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = mobilenet_v3_large.to(device)
    criterion = nn.CrossEntropyLoss()
    

    lr = 0.01 
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    print("="*29)
    print("====   START TRAINING   ====")
    print("="*29)
    
    for epoch in range(EPOCHS):
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
            
        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        bucket_correct_validation = 0
        bucket_total_validation = 0
        
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device), ages.to(device).long()
    
                preds = model(imgs)
                loss = criterion(preds, ages)
                val_loss += loss.item() * imgs.size(0)

                pred_classes = preds.argmax(dim=1)   
                bucket_correct_validation += (pred_classes == ages).sum().item()
                bucket_total_validation += ages.size(0)

        val_loss /= len(val_loader.dataset)
        
        bucket_accuracy_train = bucket_correct_train / bucket_total_train if bucket_total_train > 0 else 0
        bucket_validation_accuracy = bucket_correct_validation / bucket_total_validation if bucket_total_validation > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f} | Train Acc={bucket_accuracy_train*100:.2f}%, Val Acc={bucket_validation_accuracy*100:.2f}%")

        if bucket_validation_accuracy > top_accuracy:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/mobile_net_best.pth")
            save_confusion_matrix(model, val_loader, device)
            top_accuracy = bucket_validation_accuracy
            print(" -> New best model saved!")