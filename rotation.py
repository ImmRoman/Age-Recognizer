import cv2
import mediapipe as mp
import numpy as np
import math
import os

mp_face_mesh = mp.solutions.face_mesh

# Example input
IMAGE_FILES = ["35_ruotata.jpg"]
OUTPUT_DIR = "aligned_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        if image is None:
            print(f"Cannot open {file}")
            continue

        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print(f"No face found in {file}")
            continue

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
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)

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

        # Save
        aligned_path = os.path.join(OUTPUT_DIR, f"aligned_face_{idx}.png")
        cv2.imwrite(aligned_path, rotated_crop)
        print(f"Saved aligned and cropped face: {aligned_path}")
