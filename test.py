import numpy as np
import cv2
import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Paths
model_path = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/best_vgg16_smile_model.h5"
backup_model_path = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/vgg16_smile_model_final.h5"
attr_file = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/list_attr_celeba.csv"
preprocessed_folder = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/preprocessed_images"
test_img_folder = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/Test_images"

# Load model (try best model first, then final model)
try:
    print("Loading best model...")
    model = load_model(model_path)
    print("✓ Best model loaded successfully!")
except:
    try:
        print("Best model not found, trying final model...")
        model = load_model(backup_model_path)
        print("✓ Final model loaded successfully!")
    except:
        print("❌ No trained model found! Please run the training script first.")
        exit(1)

# Load attribute file and select test images (example: last 2000 images)
df = pd.read_csv(attr_file)
test_df = df.tail(2000)
test_imgs = test_df["image_id"].tolist()
test_labels = [1 if x == 1 else 0 for x in test_df["Smiling"]]

# Load and preprocess test images
X_test = []
y_test = []
valid_test_imgs = []

print("Loading test images...")
for i, img_name in enumerate(test_imgs):
    img_path = os.path.join(preprocessed_folder, f"{img_name}.npy")
    if os.path.exists(img_path):
        try:
            # Load preprocessed image (grayscale)
            img = np.load(img_path).reshape(64, 64, 1).astype(np.float32)
            # Convert to RGB for VGG16 (repeat grayscale across 3 channels)
            img_rgb = np.repeat(img, 3, axis=2)
            X_test.append(img_rgb)
            y_test.append(test_labels[i])
            valid_test_imgs.append(img_name)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"Loaded {len(X_test)} test images with shape: {X_test.shape}")

# Predict
print("Making predictions...")
y_pred = model.predict(X_test, verbose=0).flatten()
y_pred_labels = (y_pred > 0.5).astype(int)

# Evaluate
print(f"\n📊 Model Performance:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_labels) * 100:.2f}%")
print(f"Total test images: {len(y_test)}")
print(f"Correct predictions: {sum(y_test == y_pred_labels)}")
print(f"Incorrect predictions: {sum(y_test != y_pred_labels)}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=['Not Smiling', 'Smiling']))

# Sanity check: Predict on a few known images
print(f"\n🔍 Sample Predictions:")
sample_indices = [0, 1, 2, 3, 4]  # Change indices as needed
for idx in sample_indices:
    if idx < len(X_test):
        img = X_test[idx:idx+1]  # Keep batch dimension
        pred = model.predict(img, verbose=0)[0][0]
        pred_label = "😊 Smiling" if pred > 0.5 else "😐 Not Smiling"
        true_label = "😊 Smiling" if y_test[idx] == 1 else "😐 Not Smiling"
        confidence = max(pred, 1-pred) * 100  # Confidence score
        status = "✓" if (pred > 0.5) == (y_test[idx] == 1) else "✗"
        print(f"{status} Image: {valid_test_imgs[idx]} | Predicted: {pred_label} (Confidence: {confidence:.1f}%) | Actual: {true_label}")


def preprocess_image(img_path):
    """Preprocess a single image for VGG16 model prediction"""
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")
    # Histogram equalization
    img = cv2.equalizeHist(img)
    # Resize to 64x64
    img = cv2.resize(img, (64, 64))
    # Normalize to [0, 1]
    img = img / 255.0
    # Reshape and convert to RGB for VGG16
    img = img.reshape(64, 64, 1).astype(np.float32)
    img_rgb = np.repeat(img, 3, axis=2)  # Convert grayscale to RGB
    # Add batch dimension
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb

# Test on custom images from Test_images folder
print(f"\n🖼️  Testing on custom images:")
if os.path.exists(test_img_folder):
    # Get all image files in the test folder
    image_files = [f for f in os.listdir(test_img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_files) == 0:
        print(f"No image files found in {test_img_folder}")
    else:
        print(f"Found {len(image_files)} images to test:")
        
        for img_file in image_files:
            img_path = os.path.join(test_img_folder, img_file)
            try:
                img = preprocess_image(img_path)
                pred = model.predict(img, verbose=0)[0][0]
                pred_label = "😊 Smiling" if pred > 0.5 else "😐 Not Smiling"
                confidence = max(pred, 1-pred) * 100
                print(f"📷 {img_file} | Predicted: {pred_label} (Confidence: {confidence:.1f}%)")
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
else:
    print(f"Test images folder not found: {test_img_folder}")
    print("Create the folder and add some images to test the model on custom images.")

print(f"\n🎯 Testing completed!")
print(f"Model tested on {len(y_test)} preprocessed images with {accuracy_score(y_test, y_pred_labels) * 100:.2f}% accuracy.")