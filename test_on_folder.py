from main import plot_confusion_matrix
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split,Dataset
from Cnn import *
from database import *

FOLDER_PATH = "test_folder"
MODEL_PATH = "models\\mobilenet_v3_dataset_intero\\age_cnn_best_model.pth"  

if __name__ == "__main__":
    # Load MobileNetV3-Large pretrained on ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=8)
    model = mobilenet_v3_large.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    df = pd.DataFrame(get_data_frame(FOLDER_PATH), columns=['filepath', 'age'])
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = AgeDataset(df, transform=transform)
    DL = DataLoader(dataset, batch_size = 512)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("Using cpu to train")
        exit(-1)
    model = mobilenet_v3_large.to(device)
    print("="*39)
    print("====   ELABORAZIONE INIZIATA   ====")
    print("="*39)
    plot_confusion_matrix(model,DL,device)
