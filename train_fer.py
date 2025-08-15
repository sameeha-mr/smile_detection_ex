import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from pathlib import Path
import gc

# Emotion to Happiness Mapping (0-100%)
EMOTION_HAPPINESS_MAP = {
    0: 35,   # Angry -> 35% (was 25%)
    1: 30,   # Disgust -> 30% (was 20%)
    2: 35,   # Fear -> 35% (was 30%)
    3: 65,   # Happy -> 65% (was 70%)
    4: 30,   # Sad -> 30% (was 22%)
    5: 60,   # Surprise -> 60% (was 65%)
    6: 50    # Neutral -> 50% (unchanged)
}

EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

FOLDER_TO_EMOTION = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}

class FERDataGenerator(Sequence):
    """
    Memory-efficient data generator for FER dataset
    """
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        
        # Data augmentation
        if augment:
            self.datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                fill_mode='nearest'
            )
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch data
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            try:
                # Load and preprocess image
                image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (64, 64))
                image = image / 255.0  # Normalize to 0-1
                
                # Apply augmentation if enabled
                if self.augment:
                    image = image.reshape((1,) + image.shape)
                    image = self.datagen.flow(image, batch_size=1)[0][0]
                
                batch_images.append(image)
                batch_labels.append(self.labels[idx])
                
            except Exception as e:
                print(f"Error loading image {self.image_paths[idx]}: {e}")
                continue
        
        if len(batch_images) == 0:
            # Return dummy data if no images loaded
            batch_images = [np.zeros((64, 64, 3))]
            batch_labels = [0.5]
        
        return np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def collect_fer_paths_and_labels():
    """
    Collect image paths and labels without loading images into memory
    """
    fer_base_path = r"E:\Academic\Y4S1\CMIS 4+26 Research Project\smile_detection_ex\FER\fer2013"
    
    print(f"📊 Collecting FER-2013 image paths...")
    print(f"   Dataset path: {fer_base_path}")
    
    if not os.path.exists(fer_base_path):
        print(f"❌ Dataset not found at: {fer_base_path}")
        return None, None, None
    
    # Check for emotion folders
    possible_structures = [
        os.path.join(fer_base_path, "train"),
        fer_base_path,
        os.path.join(fer_base_path, "images")
    ]
    
    data_path = None
    for structure in possible_structures:
        if os.path.exists(structure):
            emotion_folders = [f for f in os.listdir(structure) 
                             if os.path.isdir(os.path.join(structure, f)) and 
                             f.lower() in FOLDER_TO_EMOTION.keys()]
            if emotion_folders:
                data_path = structure
                print(f"   Found emotion folders in: {data_path}")
                break
    
    if data_path is None:
        print("❌ Could not find emotion folders!")
        return None, None, None
    
    # Collect paths and labels (without loading images)
    image_paths = []
    happiness_labels = []
    emotion_labels = []
    emotion_counts = {}
    
    print("🔄 Collecting image paths...")
    
    for emotion_folder in os.listdir(data_path):
        emotion_folder_path = os.path.join(data_path, emotion_folder)
        
        if not os.path.isdir(emotion_folder_path):
            continue
        
        emotion_name = emotion_folder.lower()
        if emotion_name not in FOLDER_TO_EMOTION:
            continue
        
        emotion_id = FOLDER_TO_EMOTION[emotion_name]
        happiness_score = EMOTION_HAPPINESS_MAP[emotion_id] / 100.0  # Normalize to 0-1
        
        # Get image file paths
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(emotion_folder_path).glob(ext))
            image_files.extend(Path(emotion_folder_path).glob(ext.upper()))
        
        # Limit images per emotion to prevent memory issues
        max_images_per_emotion = 3000  # Adjust this number
        if len(image_files) > max_images_per_emotion:
            image_files = np.random.choice(image_files, max_images_per_emotion, replace=False)
        
        emotion_counts[emotion_name] = len(image_files)
        print(f"   {emotion_name}: {len(image_files)} images (→ {EMOTION_HAPPINESS_MAP[emotion_id]}% happiness)")
        
        # Add paths and labels
        for img_path in image_files:
            image_paths.append(str(img_path))
            happiness_labels.append(happiness_score)
            emotion_labels.append(emotion_id)
    
    print(f"✅ Collected {len(image_paths)} image paths!")
    print("   Final emotion distribution:")
    for emotion_name, count in emotion_counts.items():
        print(f"     {emotion_name}: {count} images")
    
    return image_paths, happiness_labels, emotion_labels

def create_happiness_model():
    """
    Create VGG16-based model for happiness regression
    """
    print("🏗️ Creating happiness prediction model...")
    
    base_model = VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    print(f"   VGG16 base loaded: {len(base_model.layers)} layers frozen")
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')  # 0-1 happiness score
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    print(f"✅ Model created with {len(model.layers)} layers")
    return model

def train_happiness_model_memory_efficient():
    """
    Memory-efficient training using data generators
    """
    print("🎯 MEMORY-EFFICIENT HAPPINESS TRAINING")
    print("="*60)
    
    # Collect paths and labels (no image loading)
    image_paths, happiness_labels, emotion_labels = collect_fer_paths_and_labels()
    if image_paths is None:
        return None
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    happiness_labels = np.array(happiness_labels)
    emotion_labels = np.array(emotion_labels)
    
    # Split data
    X_train_paths, X_val_paths, y_train, y_val, _, _ = train_test_split(
        image_paths, happiness_labels, emotion_labels,
        test_size=0.2, random_state=42, stratify=emotion_labels
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {len(X_train_paths)} samples")
    print(f"   Validation: {len(X_val_paths)} samples")
    print(f"   Train happiness range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"   Val happiness range: {y_val.min():.2f} - {y_val.max():.2f}")
    
    # Create data generators
    batch_size = 16  # Smaller batch size for memory efficiency
    train_generator = FERDataGenerator(
        X_train_paths, y_train, 
        batch_size=batch_size, 
        shuffle=True, 
        augment=True
    )
    val_generator = FERDataGenerator(
        X_val_paths, y_val, 
        batch_size=batch_size, 
        shuffle=False, 
        augment=False
    )
    
    print(f"   Batch size: {batch_size}")
    print(f"   Training batches: {len(train_generator)}")
    print(f"   Validation batches: {len(val_generator)}")
    
    # Create model
    model = create_happiness_model()
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            verbose=1, 
            min_lr=1e-7
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True, 
            verbose=1
        ),
        ModelCheckpoint(
            'happiness_model_memory_efficient.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training parameters
    epochs = 15  # Reduced for faster training
    
    print(f"🚀 Starting memory-efficient training...")
    print(f"   Epochs: {epochs}")
    print(f"   Using data generators: Yes")
    print(f"   Memory usage: Low (batch-by-batch loading)")
    
    # Train with generators
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('happiness_model_memory_final.h5')
    
    # Evaluate model on a small validation subset
    print("\n📊 Model Evaluation:")
    val_sample = val_generator[0]  # Get first batch
    val_loss, val_mae, val_mse = model.evaluate(val_sample[0], val_sample[1], verbose=0)
    print(f"   Sample Validation Loss (MSE): {val_loss:.4f}")
    print(f"   Sample Mean Absolute Error: {val_mae:.4f}")
    print(f"   Sample Root Mean Squared Error: {np.sqrt(val_mse):.4f}")
    
    # Test predictions
    print("\n🧪 Sample Predictions:")
    predictions = model.predict(val_sample[0][:5], verbose=0)
    for i in range(min(5, len(predictions))):
        actual = val_sample[1][i] * 100
        predicted = predictions[i][0] * 100
        print(f"   Sample {i+1}: Actual: {actual:.1f}% | Predicted: {predicted:.1f}% | Diff: {abs(actual-predicted):.1f}%")
    
    # Clear memory
    del train_generator, val_generator
    gc.collect()
    
    print("✅ Memory-efficient training completed!")
    return model, history

def test_memory_efficient_model():
    """
    Test the memory-efficient trained model
    """
    print("🧪 Testing Memory-Efficient Model...")
    
    try:
        from tensorflow.keras.models import load_model
        model = load_model('happiness_model_memory_efficient.h5')
        print("✅ Model loaded successfully!")
        
        # Test with random images
        test_images = np.random.random((3, 64, 64, 3)).astype(np.float32)
        predictions = model.predict(test_images, verbose=0)
        
        print("📊 Test Results:")
        for i, pred in enumerate(predictions):
            happiness_percent = pred[0] * 100
            print(f"   Test Image {i+1}: {happiness_percent:.1f}% happiness")
        
        print("✅ Memory-efficient model is working!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    print("🎯 Memory-Efficient FER Happiness Training")
    print("="*60)
    print("⚡ Using data generators to prevent memory overflow")
    print("📊 Batch-by-batch loading instead of loading all images")
    print("="*60)
    
    # Train the model
    model, history = train_happiness_model_memory_efficient()
    
    # Test the trained model
    if model:
        test_memory_efficient_model()
        print("\n🎉 Memory-Efficient Training Complete!")
        print("📁 Models saved:")
        print("   - happiness_model_memory_efficient.h5 (best model)")
        print("   - happiness_model_memory_final.h5 (final model)")
        print("\n💡 Key improvements:")
        print("   ✅ No memory overflow (uses data generators)")
        print("   ✅ Handles large datasets efficiently")
        print("   ✅ Smaller batch size (16 vs 32)")
        print("   ✅ Limited images per emotion (3000 max)")
    else:
        print("❌ Training failed.")