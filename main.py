
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
import torchvision.models as models


from Cnn import *
from database import *
top_accuracy = 0
if __name__ == "__main__":
    # Load MobileNetV3-Large pretrained on ImageNet
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained = True)
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=8)

    # Load MobileNetV3-Small pretrained on ImageNet
    # mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    # Create a Pandas DataFrame
    df = pd.DataFrame(get_data_frame("cropped_dataset"), columns=['filepath', 'age'])
    # df_validation = pd.DataFrame(get_data_frame("validation_cropped_faces"), columns=['filepath', 'age'])
    rotations_iterator = [0]
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = AgeDataset(df, transform=transform)
    # validation = AgeDataset(df_validation,transform=transform)

    # Train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size = 256, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size = 64)

    val_loader = DataLoader(val_ds, batch_size = 64)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using cpu to train")
        exit(-1)
    model = mobilenet_v3_large.to(device)
    # UTKFace approximate bucket distribution (example, adjust if you have exact counts):
    # [0-3]: 6000, [4-7]: 5000, [8-14]: 7000, [15-21]: 8000, [22-37]: 12000, [38-47]: 6000, [48-59]: 4000, [60+]: 3000
    # bucket_counts = torch.tensor([6000, 5000, 7000, 8000, 12000, 6000, 4000, 3000], dtype=torch.float)
    # bucket_weights = bucket_counts.sum() / (len(bucket_counts) * bucket_counts)
    # bucket_weights = bucket_weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=bucket_weights)
    criterion = nn.CrossEntropyLoss()

    lr = 0.001  # Initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)



    print("="*29)
    print("====   INIZIO TRAINING   ====")
    print("="*29)
    # Training loop
    for epoch in range(120):
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

        # Update lr variable to reflect current optimizer lr
        lr = optimizer.param_groups[0]['lr']
        
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
        
    
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, lr={lr}")
        bucket_accuracy_validation = bucket_correct_validation / bucket_total_validation if bucket_total_validation > 0 else 0
        if bucket_accuracy_validation > top_accuracy:
            torch.save(model.state_dict(), f"models/age_cnn_best_model.pth")
            top_accuracy = bucket_accuracy_validation
        #Compute accuracies
        print(f" Testing bucket accuracy : {bucket_accuracy_train*100:.2f}%")
        print(f" Validation bucket accuracy : {bucket_accuracy_validation*100:.2f}%")

        if ((epoch + 10) % 30 == 0):
            # Save model at the end of each epoch
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/age_cnn_epoch_{epoch+1}.pth")
            plot_confusion_matrix(model,val_loader , device)


