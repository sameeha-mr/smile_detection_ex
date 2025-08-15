import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16

# File paths
attr_file = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/list_attr_celeba.csv"
preprocessed_folder = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/preprocessed_images"

# Load attributes file
df = pd.read_csv(attr_file)

# Get lists of image IDs based on the "Smiling" attribute
smiling_images = df[df["Smiling"] == 1]["image_id"].tolist()
non_smiling_images = df[df["Smiling"] == -1]["image_id"].tolist()

# Create combined lists; for smiling set label 1, for non-smiling set label 0
all_images = smiling_images + non_smiling_images
all_labels = [1] * len(smiling_images) + [0] * len(non_smiling_images)

# Split the dataset (80% training, 20% validation)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)

# Improved Preprocessing with Histogram Equalization
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Avoid errors if the image loading fails
    
    img = cv2.equalizeHist(img)  # Improve contrast
    img = cv2.resize(img, (64, 64)) / 255.0  # Resize and normalize
    return img.reshape(64, 64, 1).astype(np.float32)  # Ensure float32

# Enhanced Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

def data_generator(image_names, labels, batch_size, folder):
    num_samples = len(image_names)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_images = []
            batch_labels = []
            for i in indices[start:end]:
                img_name = image_names[i]
                label = labels[i]
                img_path = os.path.join(folder, f"{img_name}.npy")
                if os.path.exists(img_path):
                    try:
                        img = np.load(img_path).reshape(64, 64, 1).astype(np.float32)
                        # Convert grayscale to RGB for VGG16 (repeat channel 3 times)
                        img = np.repeat(img, 3, axis=2)
                        batch_images.append(img)
                        batch_labels.append(label)
                    except Exception as e:
                        print(f"Error loading {img_name}: {e}")
                        continue
            if batch_images:
                try:
                    yield datagen.flow(np.array(batch_images), np.array(batch_labels), batch_size=len(batch_images)).__next__()
                except Exception as e:
                    print(f"Error in data generator: {e}")
                    # Fallback: return raw data without augmentation
                    yield np.array(batch_images), np.array(batch_labels)

# Define VGG16 Transfer Learning Model
print("Loading VGG16 base model...")
try:
    # Load VGG16 base model (without top classification layer)
    base_model = VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    print("VGG16 loaded successfully!")
    
    # Freeze the base model layers
    base_model.trainable = False
    print(f"VGG16 has {len(base_model.layers)} layers, all frozen.")
    
    # Add custom classification layers on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    print(f"Complete model has {len(model.layers)} layers")
    print("Model architecture created successfully!")
    
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Enhanced Learning Rate and Callbacks
opt = Adam(learning_rate=0.0001)  # Lower learning rate for transfer learning
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(
    'E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/best_vgg16_smile_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)

print("Compiling model...")
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
print("Model compiled successfully!")

# Train the Model with Augmented Data
batch_size = 32  # Reduced batch size to prevent memory issues
steps_per_epoch = len(train_imgs) // batch_size
validation_steps = len(val_imgs) // batch_size

print(f"Training images: {len(train_imgs)}")
print(f"Validation images: {len(val_imgs)}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Batch size: {batch_size}")

train_gen = data_generator(train_imgs, train_labels, batch_size, preprocessed_folder)
val_gen = data_generator(val_imgs, val_labels, batch_size, preprocessed_folder)

# Test generators before training
print("Testing data generators...")
try:
    train_batch = next(train_gen)
    print(f"Train batch shape: {train_batch[0].shape}, {train_batch[1].shape}")
    val_batch = next(val_gen)
    print(f"Validation batch shape: {val_batch[0].shape}, {val_batch[1].shape}")
    print("Data generators working correctly!")
except Exception as e:
    print(f"Error with data generators: {e}")
    exit(1)

print("Starting model training...")
try:
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=20,  # Increased epochs for better training
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint],
        verbose=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()

# Save Model
model.save("E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/vgg16_smile_model.h5")
print("VGG16 Transfer Learning model training completed and saved.")

# Improved Smile Detection Function for VGG16 Model
def detect_smile(img_name):
    from tensorflow.keras.models import load_model
    model = load_model("E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/vgg16_smile_model.h5")
    
    img_path = os.path.join(preprocessed_folder, f"{img_name}.npy")
    if not os.path.exists(img_path):
        return "Error: Image not found"
    
    img = np.load(img_path).reshape(1, 64, 64, 1).astype(np.float32)
    # Convert grayscale to RGB for VGG16
    img = np.repeat(img, 3, axis=3)
    prediction = model.predict(img)[0][0]
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    return f"Prediction for {img_name}: {'Smiling' if prediction > 0.5 else 'Not Smiling'} (Confidence: {confidence:.2f}%)"

# Test the Detection Function
print(detect_smile("000001.jpg"))
print(detect_smile("000006.jpg"))
print(detect_smile("000008.jpg"))
print(detect_smile("000025.jpg"))