from database import save_confusion_matrix
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from Cnn import *
from database import *
import torch

FOLDER_PATH = "validation_cropped_faces"
MODEL_PATH = "models\\mobile_net_best_98.pth"  

if __name__ == "__main__":
    # Load MobileNetV3-Large pretrained on ImageNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=8)
    model = mobilenet_v3_large.to(device)
    # model = AgeCNN()
    # model = model.to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    df = pd.DataFrame(get_data_frame(FOLDER_PATH), columns=['filepath', 'age'])
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = AgeDataset(df, transform=transform)
    # Load the datase
    dataset = torch.load("validation_dataset_mobile.pth", weights_only=False)
    DL = DataLoader(dataset, batch_size=512)



    print("="*39)
    print("====   ELABORAZIONE INIZIATA   ====")
    print("="*39)
    save_confusion_matrix(model,DL,device)
