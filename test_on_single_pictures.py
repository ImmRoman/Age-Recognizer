import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from Cnn import get_age_range

MODEL_PATH = "models\\mobilenet_v3_dataset_intero\\age_cnn_best_model.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=8)
model = mobilenet_v3_large.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def open_image():
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not filepath:
        return

    image = Image.open(filepath).convert("RGB")
    display_img = image.resize((300, 300))
    tk_img = ImageTk.PhotoImage(display_img)

    img_label.configure(image=tk_img)
    img_label.image = tk_img

    # Prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        predicted_label = get_age_range(predicted_class)

    text_label.configure(text=f"Predicted age range: {predicted_label}")


window = tk.Tk()
window.title("Age Prediction GUI")

open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack(pady=10)

img_label = tk.Label(window)
img_label.pack()

text_label = tk.Label(window, text="", font=("Arial", 16))
text_label.pack(pady=10)

window.mainloop()
