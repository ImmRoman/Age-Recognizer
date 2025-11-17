
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

DATABASE_PATH = "cropped_output"

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
    df = pd.DataFrame(get_data_frame(DATABASE_PATH), columns=['filepath', 'age'])
    transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    # torch.save(train_loader.dataset, 'train_dataset.pth')
    # val_loader = DataLoader(val_ds, batch_size = 64)

    val_loader = DataLoader(val_ds, batch_size = 64)
    # torch.save(val_loader.dataset, 'validation_dataset.pth')
    
    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using cpu to train")
        exit(-1)
    
    model = mobilenet_v3_large.to(device)
    # model = AgeCNN().to(device)

    bucket_counts = torch.tensor([3765, 1419, 1934, 2508, 12725, 3274, 3701, 4140], dtype=torch.float)
    bucket_weights = bucket_counts.sum() / (len(bucket_counts) * bucket_counts)
    bucket_weights = bucket_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=bucket_weights)
    # criterion = nn.CrossEntropyLoss()
    
    lr = 0.001  # Initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # import matplotlib.pyplot as plt
    # def plot_class_distribution(df):
    #     class_counts = df['age'].value_counts().sort_index()
    #     plt.figure(figsize=(10, 6))
    #     ax = class_counts.plot(kind='bar')
    #     plt.title('Distribution of Age Classes')
    #     plt.xlabel('Age Class')
    #     plt.ylabel('Count')
        
    #     # Add value labels on top of each bar
    #     for i, v in enumerate(class_counts):
    #         ax.text(i, v, str(v), ha='center', va='bottom')
        
    #     plt.xticks(rotation=0)
    #     plt.grid(axis='y')
    #     plt.show()

    # plot_class_distribution(df)


    print("="*29)
    print("====   INIZIO TRAINING   ====")
    print("="*29)
    # Training loop
    for epoch in range(1200):
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
        
    
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f} ")
        bucket_accuracy = bucket_correct_train / bucket_total_train if bucket_total_train > 0 else 0
        bucket_validation_accuracy = bucket_correct_validation / bucket_total_validation if bucket_total_validation > 0 else 0
        if bucket_validation_accuracy > top_accuracy:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/age_cnn_weighted_with_arg.pth")
            top_accuracy = bucket_validation_accuracy
        #Compute accuracies
        print(f" Testing bucket accuracy : {bucket_accuracy*100:.2f}%")
        print(f" Validation bucket accuracy : {bucket_validation_accuracy*100:.2f}%")


        if ((epoch + 1) % 50 == 0):
            # Save model at the end of each epoch
            # os.makedirs("models", exist_ok=True)
            # torch.save(model.state_dict(), f"models/age_weighted_cnn_epoch_{epoch+1}_{bucket_accuracy:.2f}.pth")
            plot_confusion_matrix(model,val_loader , device)


