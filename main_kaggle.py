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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image



# KAGGLE PATH - automatically detects if running on Kaggle
if os.path.exists('/kaggle/input'):
    # Running on Kaggle - adjust this to match your dataset name
    DATABASE_PATH = "/kaggle/input/age-recognizer/cropped_dataset"
else:
    # Running locally
    DATABASE_PATH = "cropped_output"

print(f"Using database path: {DATABASE_PATH}")


def get_age_bucket(age):
    match age:
        case a if 0<=a<=3:
            return 0
        case a if 4<=a<=7:
            return 1
        case a if 8<=a<=14:
            return 2
        case a if 15<=a<=21:
            return 3
        case a if 22<=a<=37:
            return 4
        case a if 38<=a<=47:
            return 5
        case a if 48<=a<=59:
            return 6
        case a if a>=60:
            return 7
        
def get_age_range(bucket: int) -> str:
    match bucket:
        case 0: return "0-3"
        case 1: return "4-7"
        case 2: return "8-14"
        case 3: return "15-21"
        case 4: return "22-37"
        case 5: return "38-47"
        case 6: return "48-59"
        case 7: return "60+"
        case _: return "Unknown"



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
        img = img.resize(size=(224,224))
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
        if filename.endswith('.png') or filename.endswith('.jpg'):
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
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=100)
    plt.close()


# --- Main Execution ---

top_accuracy = 0

# Setup MobileNet with updated syntax
from torchvision.models import MobileNet_V3_Large_Weights
weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
mobilenet_v3_large = models.mobilenet_v3_large(weights=weights)

# Adjust classifier for 8 classes
_in_features = None
for m in mobilenet_v3_large.classifier.modules():
    if isinstance(m, nn.Linear):
        _in_features = m.in_features
        break
if _in_features is None: _in_features = 1024

mobilenet_v3_large.classifier = nn.Sequential(
    nn.Linear(_in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(512, 8)
)

# Freeze backbone
for name, param in mobilenet_v3_large.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Initialize weights
for m in mobilenet_v3_large.classifier.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

print(f"Transfer learning setup: frozen backbone, trainable params = {sum(p.numel() for p in mobilenet_v3_large.parameters() if p.requires_grad)}")

# Load Data
print(f"Looking for data in: {os.path.abspath(DATABASE_PATH)}")
raw_data = get_data_frame(DATABASE_PATH)
if len(raw_data) == 0:
    print("ERROR: No images found. Check your database path.")
    print(f"Contents of parent directory: {os.listdir(os.path.dirname(DATABASE_PATH)) if os.path.exists(os.path.dirname(DATABASE_PATH)) else 'Path does not exist'}")
else:
    print(f"Found {len(raw_data)} images")
    df = pd.DataFrame(raw_data, columns=['filepath', 'age'])
    
    # Transforms
    transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = AgeDataset(df, transform=transform)

    # Split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)  # Reduced workers for Kaggle
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=2)

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
    
    EPOCHS = 300 
    
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
            # Save to /kaggle/working on Kaggle (only writeable directory)
            save_dir = "/kaggle/working" if os.path.exists('/kaggle/working') else "models"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/mobile_net_best.pth")
            save_confusion_matrix(model, val_loader, device)
            top_accuracy = bucket_validation_accuracy
            print(f" -> New best model saved to {save_dir}!")
    
    print(f"\nTraining complete! Best validation accuracy: {top_accuracy*100:.2f}%")