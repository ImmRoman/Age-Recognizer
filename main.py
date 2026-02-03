import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as T
import torch.optim as optim
import torchvision.models as models

DATABASE_PATH = "cropped_output"

from Cnn import *
from database import *    

top_accuracy = 0

if __name__ == "__main__":


    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=2)   

    df = pd.DataFrame(get_data_frame(DATABASE_PATH), columns=['filepath', 'gender']) 

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = GenderDataset(df, transform=transform)   



    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using CPU to train → EXIT")
        exit(-1)

    model = mobilenet_v3_large.to(device)

 
    class_counts = torch.tensor([1, 1], dtype=torch.float)  
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)   

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)



    print("=" * 39)
    print("====   START TRAINING (GENDER)   ====")
    print("=" * 39)

    for epoch in range(1200):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for imgs, genders in train_loader:
            imgs, genders = imgs.to(device), genders.to(device).long()

            optimizer.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, genders)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct_train += (preds.argmax(dim=1) == genders).sum().item()
            total_train += genders.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Validation -----------------------------------------------------
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, genders in val_loader:
                imgs, genders = imgs.to(device), genders.to(device).long()
                preds = model(imgs)

                loss = criterion(preds, genders)
                val_loss += loss.item() * imgs.size(0)

                correct_val += (preds.argmax(dim=1) == genders).sum().item()
                total_val += genders.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val

        print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")
        print(f"Train accuracy: {train_acc*100:.2f}% | Val accuracy: {val_acc*100:.2f}%")

        # Save best model ------------------------------------------------
        if val_acc > top_accuracy:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/mobile_net_gender_detection.pth")
            top_accuracy = val_acc
            save_confusion_matrix(model, val_loader, device)

