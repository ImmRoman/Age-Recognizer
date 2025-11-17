import cv2
import os
import glob

def crop_faces(image_path, output_folder="cropped_faces", box_scale=1.2):
    # Load Haar cascade for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("No faces found in", image_path)
        return

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Crop and save faces
    for i, (x, y, w, h) in enumerate(faces, 1):
        # Optionally scale the bounding box
        x1 = max(int(x - w*(box_scale-1)/2), 0)
        y1 = max(int(y - h*(box_scale-1)/2), 0)
        x2 = min(int(x + w*(1 + (box_scale-1)/2)), img.shape[1])
        y2 = min(int(y + h*(1 + (box_scale-1)/2)), img.shape[0])

        face_crop = img[y1:y2, x1:x2]
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        save_path = os.path.join(output_folder, f"{name}_face{i}{ext}")
        cv2.imwrite(save_path, face_crop)
        print(f"Saved cropped face to {save_path}")

# Example: crop all jpg images in a folder
for image_file in glob.glob("dataset/*.jpg"):
    crop_faces(image_file)
