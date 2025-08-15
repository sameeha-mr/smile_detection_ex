import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import skfuzzy as fuzz
from skfuzzy import control as ctrl
cnn_model = load_model("E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/smile_model.h5")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
from Backup.fuzzy import compute_happiness
from Backup.extract_features import extract_features

def preprocess_image_for_cnn(img_path):
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img.reshape(1, 64, 64, 1).astype(np.float32)

def get_cnn_smile_score_from_raw(img_path):
    img = preprocess_image_for_cnn(img_path)
    return float(cnn_model.predict(img)[0][0])

def predict_happiness_for_image(img_path):
    cnn_smile_score = get_cnn_smile_score_from_raw(img_path)
    features = extract_features(img_path, cnn_smile_score)
    print("Extracted features:", features)
    if features is None:
        print("Face not detected or feature extraction failed.")
        return None
    score = compute_happiness(features)
    print(f"Happiness score for {img_path}: {score:.2f}")
    return score

if __name__ == "__main__":
    img_path = "E:\Academic\Y4S1\CMIS 4+26 Research Project\smile_detection_ex\download.jpg"  # <-- Change this to your image
    predict_happiness_for_image(img_path)
    
