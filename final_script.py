import cv2
import mediapipe as mp
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import math
from Cnn import get_age_range

MODEL_PATH = "models\\age_cnn_best_model.pth"  
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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging


def open_camera():
    # Initialize the camera (0 is usually the default webcam)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5) as face_mesh:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
                
            
        # Convert the BGR image to RGB before processing
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                cv2.imshow('Camera Feed', frame)
                print("No face landmarks found")
                continue
            h, w, _ = frame.shape
            face_landmarks = results.multi_face_landmarks[0]

            # Get landmark points for both eyes (approximate)
            left_eye = face_landmarks.landmark[33]    # left eye outer corner
            right_eye = face_landmarks.landmark[263]  # right eye outer corner

            # Convert to pixel coords
            left_eye = np.array([left_eye.x * w, left_eye.y * h])
            right_eye = np.array([right_eye.x * w, right_eye.y * h])

            # Compute the rotation angle
            dx, dy = right_eye - left_eye
            angle = math.degrees(math.atan2(dy, dx))

            # Compute face center for rotation
            center_x = int((left_eye[0] + right_eye[0]) / 2)
            center_y = int((left_eye[1] + right_eye[1]) / 2)

            # Rotate the image around the eyes' midpoint
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_image = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
            if rotated_image is not None:
                cv2.imshow('Rotated Image', rotated_image)
            # Recompute bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min = int(max(min(xs) * w, 0))
            y_min = int(max(min(ys) * h, 0))
            x_max = int(min(max(xs) * w, w))
            y_max = int(min(max(ys) * h, h))

            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, w)
            y_max = min(y_max + pad, h)

            # Crop from the rotated image
            rotated_crop = rotated_image[y_min:y_max, x_min:x_max]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Prediction
            img_tensor = transform(Image.fromarray(rotated_crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                predicted_label = get_age_range(predicted_class)
            cv2.putText(frame, f"Age: {predicted_label}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


            # Display the frame
            cv2.imshow('Camera Feed', frame)
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the camera and destroy windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()