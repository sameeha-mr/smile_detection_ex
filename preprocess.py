import cv2
import numpy as np
import pandas as pd
import os

# File paths
attr_file = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/list_attr_celeba.csv"
img_folder = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/img_align_celeba"
preprocessed_folder = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/preprocessed_images"

# Ensure output folder exists
os.makedirs(preprocessed_folder, exist_ok=True)

# Load attributes file
df = pd.read_csv(attr_file)
smiling_images = df[df["Smiling"] == 1]["image_id"].tolist()
non_smiling_images = df[df["Smiling"] == -1]["image_id"].tolist()

# Function to preprocess and save images
def preprocess_and_save(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load: {img_path}")
        return False
    
    img = cv2.equalizeHist(img)  # Improve contrast
    img = cv2.resize(img, (64, 64)) / 255.0
    np.save(save_path, img)  # Save as numpy array (.npy)
    return True

# Process and save images
for img_name in smiling_images + non_smiling_images:
    img_path = os.path.join(img_folder, img_name)
    save_path = os.path.join(preprocessed_folder, f"{img_name}.npy")

    success = preprocess_and_save(img_path, save_path)
    if success:
        print(f"Saved: {save_path}")

print("Preprocessing completed. Images saved in:", preprocessed_folder)