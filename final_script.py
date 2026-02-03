import cv2
import mediapipe as mp
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import numpy as np
import math
from Cnn import get_age_range

AGE_MODEL_PATH = "models\\Eta\\MobileNetV3\\normal\\mobile_net_best.pth"  
GENDER_MODEL_PATH = "models\\Genere\\MovileNetV2_weight0.5\\mobilev2_weight.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Age Model
mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
mobilenet_v3_large.classifier = nn.Sequential(
    nn.Linear(960, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.4),
    nn.Linear(512, 8)
)
# Gender Model
model = mobilenet_v3_large.to(device)
model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=device, weights_only=False))
model.eval()

mobilenet_v2_gender = models.mobilenet_v2(weights=None, width_mult=0.5)
in_features_gender = mobilenet_v2_gender.classifier[1].in_features

mobilenet_v2_gender.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features_gender, 1) 
)

model_gender = mobilenet_v2_gender.to(device)
model_gender.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=device, weights_only=False))
model_gender.eval()

# Transform with normalization
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess_batch(crops):
    """Batch preprocessing for multiple faces"""
    tensors = []
    for crop in crops:
        img_pil = Image.fromarray(crop)
        tensor = transform(img_pil)
        tensors.append(tensor)
    return torch.stack(tensors).to(device)


def predict_batch(img_tensors):
    """Batch prediction for better GPU utilization"""
    with torch.no_grad():
        # Age predictions
        age_outputs = model(img_tensors)
        age_classes = torch.argmax(age_outputs, dim=1)

        gender_outputs = model_gender(img_tensors)
        gender_probs = torch.sigmoid(gender_outputs)
        gender_classes = (gender_probs > 0.5).long().squeeze(1)
    
    return age_classes, gender_classes


def open_camera():
    """Open camera and detect age/gender in real-time with GPU optimization"""
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened. Press 'q' to quit.")
    
    # Frame counter
    frame_count = 0
    import time
    start_time = time.time()
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            h, w, _ = frame.shape
            
            crops = []
            face_data = []
            
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                
                # Convert to pixel coords
                left_eye_px = np.array([left_eye.x * w, left_eye.y * h])
                right_eye_px = np.array([right_eye.x * w, right_eye.y * h])
                
                # Compute rotation angle
                dx, dy = right_eye_px - left_eye_px
                angle = math.degrees(math.atan2(dy, dx))
                
                # Compute face center
                center_x = int((left_eye_px[0] + right_eye_px[0]) / 2)
                center_y = int((left_eye_px[1] + right_eye_px[1]) / 2)
                
                # Rotate image
                rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
                
                # Get bounding box
                xs = [lm.x for lm in face_landmarks.landmark]
                ys = [lm.y for lm in face_landmarks.landmark]
                x_min = int(max(min(xs) * w, 0))
                y_min = int(max(min(ys) * h, 0))
                x_max = int(min(max(xs) * w, w))
                y_max = int(min(max(ys) * h, h))
                
                # Add padding
                pad = 20
                x_min = max(x_min - pad, 0)
                y_min = max(y_min - pad, 0)
                x_max = min(x_max + pad, w)
                y_max = min(y_max + pad, h)
                
                # Crop face
                if y_max > y_min and x_max > x_min:
                    rotated_crop = rotated_frame[y_min:y_max, x_min:x_max]
                    crops.append(rotated_crop)
                    face_data.append({
                        'rotated_frame': rotated_frame,
                        'bbox': (x_min, y_min, x_max, y_max)
                    })
            
            # Batch prediction for all faces
            if len(crops) > 0:
                try:
                    # Preprocess all crops at once
                    img_tensors = preprocess_batch(crops)
                    
                    # Predict in batch
                    age_classes, gender_classes = predict_batch(img_tensors)
                    
                    # Draw results on each face
                    for i, data in enumerate(face_data):
                        rotated_frame = data['rotated_frame']
                        x_min, y_min, x_max, y_max = data['bbox']
                        
                        # Get predictions
                        predicted_age_class = age_classes[i].item()
                        predicted_label = get_age_range(predicted_age_class)
                        
                        predicted_gender_class = gender_classes[i].item()
                        predicted_gender = "M" if predicted_gender_class == 0 else "F"
                        
                        # Draw rectangle and text
                        cv2.rectangle(rotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        text = f"Age: {predicted_label} | Gender: {predicted_gender}"
                        cv2.putText(rotated_frame, text, 
                                   (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (36, 255, 12), 2)
                        
                        # Use the last processed frame for display
                        frame = rotated_frame
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Display frame
            cv2.imshow('Camera Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    print(f"Camera closed. Average FPS: {fps:.1f}")


if __name__ == "__main__":
    open_camera()