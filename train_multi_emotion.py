import os
import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy  # Import for top-k accuracy
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

# Emotion labels
EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

FOLDER_TO_EMOTION = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}

def create_improved_multi_emotion_model():
    """Create IMPROVED multi-emotion model with better architecture"""
    print("🏗️ Creating IMPROVED multi-emotion classification model")
    
    # Base model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(64, 64, 3)
    )
    
    # Fine-tune last few layers
    for layer in base_model.layers[:-6]:  # Freeze more layers
        layer.trainable = False
    
    # IMPROVED architecture
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        
        # First block with batch normalization
        Dense(512, activation='relu'),
        BatchNormalization(),  # Helps with training stability
        Dropout(0.4),          # Higher dropout
        
        # Second block
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third block (smaller)
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Output layer - CRITICAL: Use softmax for classification
        Dense(7, activation='softmax', name='emotion_output')
    ])
    
    # FIXED compilation - remove problematic top_2_accuracy
    model.compile(
        optimizer=Adam(
            learning_rate=0.0005,  # Lower learning rate for stability
            beta_1=0.9,
            beta_2=0.999
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]  # FIXED: Use proper import
    )
    
    model.summary()
    return model

def create_data_augmentation():
    """Create data augmentation for better generalization"""
    
    # IMPROVED augmentation - FIXED: Remove rescale (we'll normalize separately)
    datagen = ImageDataGenerator(
        rotation_range=20,           # Rotate images
        width_shift_range=0.1,       # Shift horizontally
        height_shift_range=0.1,      # Shift vertically
        horizontal_flip=True,        # Mirror images
        zoom_range=0.1,             # Zoom in/out
        brightness_range=[0.8, 1.2], # Brightness variation
        fill_mode='nearest'          # Fill new pixels
        # NOTE: rescale removed - we normalize in preprocessing
    )
    
    return datagen

def collect_balanced_emotion_data(data_path, max_per_emotion=500):
    """Collect BALANCED emotion data"""
    print(f"🔍 Collecting BALANCED emotion data from: {data_path}")
    
    # Check if path exists
    if not os.path.exists(data_path):
        print(f"❌ Dataset path not found: {data_path}")
        
        # Try alternative paths
        alternative_paths = [
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013",
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/fer2013",
            "./FER/fer2013/train",
            "./fer2013/train"
        ]
        
        print("🔍 Trying alternative paths:")
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"   ✅ Found: {alt_path}")
                data_path = alt_path
                break
            else:
                print(f"   ❌ Not found: {alt_path}")
        
        if not os.path.exists(data_path):
            print("💡 Please check your dataset path!")
            return [], []
    
    image_paths = []
    emotion_labels = []
    emotion_counts = {}
    
    # First pass: count all images
    emotion_files = {}
    for emotion_name, emotion_id in FOLDER_TO_EMOTION.items():
        emotion_folder = os.path.join(data_path, emotion_name)
        
        if os.path.exists(emotion_folder):
            files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                files.extend(list(Path(emotion_folder).glob(ext)))
                files.extend(list(Path(emotion_folder).glob(ext.upper())))
            
            emotion_files[emotion_name] = files
            print(f"   Found {len(files)} {emotion_name} images")
        else:
            print(f"   ⚠️ Missing folder: {emotion_folder}")
            emotion_files[emotion_name] = []
    
    # Check if we found any images
    total_found = sum(len(files) for files in emotion_files.values())
    if total_found == 0:
        print("❌ No images found in any emotion folder!")
        return [], []
    
    # Find minimum count for balancing (excluding empty folders)
    non_empty_counts = [len(files) for files in emotion_files.values() if len(files) > 0]
    if not non_empty_counts:
        print("❌ No valid emotion folders found!")
        return [], []
    
    min_count = min(non_empty_counts)
    balanced_count = min(min_count, max_per_emotion)
    
    print(f"\n⚖️ Balancing dataset: using {balanced_count} images per emotion")
    
    # Second pass: collect balanced data
    for emotion_name, files in emotion_files.items():
        if len(files) > 0:
            emotion_id = FOLDER_TO_EMOTION[emotion_name]
            
            # Randomly sample for balance
            if len(files) > balanced_count:
                selected_files = random.sample(files, balanced_count)
            else:
                selected_files = files
            
            emotion_counts[emotion_name] = len(selected_files)
            
            for img_path in selected_files:
                image_paths.append(str(img_path))
                emotion_labels.append(emotion_id)
    
    print(f"✅ Collected {len(image_paths)} BALANCED samples")
    print("   Final distribution:", emotion_counts)
    
    return image_paths, emotion_labels

def preprocess_image_improved(image_path):
    """IMPROVED image preprocessing"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with better interpolation
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1 (IMPORTANT: Do this here, not in augmentation)
        img = img / 255.0
        
        return img.astype(np.float32)
        
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def train_improved_multi_emotion_model(data_path):
    """Train IMPROVED multi-emotion model"""
    print("🚀 TRAINING IMPROVED MULTI-EMOTION MODEL")
    print("="*60)
    
    # Collect balanced data
    image_paths, emotion_labels = collect_balanced_emotion_data(data_path, max_per_emotion=400)
    
    if len(image_paths) == 0:
        print("❌ No training data found!")
        return None, None
    
    # Preprocess images
    print("🔄 Preprocessing images...")
    images = []
    labels = []
    
    for i, (img_path, emotion_id) in enumerate(zip(image_paths, emotion_labels)):
        if i % 200 == 0:
            print(f"   Processed {i}/{len(image_paths)} images...")
        
        img = preprocess_image_improved(img_path)
        if img is not None:
            images.append(img)
            labels.append(emotion_id)
    
    if len(images) == 0:
        print("❌ No valid images after preprocessing!")
        return None, None
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert to categorical
    labels_categorical = to_categorical(labels, num_classes=7)
    
    print(f"✅ Preprocessed {len(images)} images")
    print(f"   Shape: {images.shape}")
    print(f"   Labels: {labels_categorical.shape}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n📊 Class distribution:")
    for emotion_id, count in zip(unique, counts):
        emotion_name = EMOTION_LABELS[emotion_id]
        print(f"   {emotion_name}: {count} samples")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_categorical, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Create improved model
    model = create_improved_multi_emotion_model()
    
    # Improved callbacks
    callbacks = [
        ModelCheckpoint(
            'multi_emotion_model_IMPROVED.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,        # Reduce to 50%
            patience=3,        # Wait 3 epochs
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,       # Wait 10 epochs
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Create data augmentation
    datagen = create_data_augmentation()
    
    # IMPROVED training with augmentation
    print("🏃 Starting IMPROVED training with data augmentation...")
    
    try:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=30,                    # More epochs
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        
        # Fallback: Train without augmentation
        print("🔄 Trying training WITHOUT augmentation...")
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=30,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        except Exception as e2:
            print(f"❌ Training failed completely: {e2}")
            return None, None
    
    # Evaluate final model
    print("\n📊 Final Evaluation:")
    try:
        evaluation = model.evaluate(X_val, y_val, verbose=0)
        val_loss = evaluation[0]
        val_accuracy = evaluation[1]
        
        print(f"   Validation Accuracy: {val_accuracy:.3f}")
        print(f"   Validation Loss: {val_loss:.3f}")
        
        if len(evaluation) > 2:  # If top-2 accuracy is available
            val_top2 = evaluation[2]
            print(f"   Validation Top-2 Accuracy: {val_top2:.3f}")
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
    
    # Save final model
    model.save('multi_emotion_model_IMPROVED_final.h5')
    print("✅ IMPROVED model saved!")
    
    # Test with some predictions
    print(f"\n🧪 Testing model predictions:")
    try:
        test_predictions = model.predict(X_val[:5], verbose=0)
        
        for i, pred in enumerate(test_predictions):
            true_emotion_id = np.argmax(y_val[i])
            pred_emotion_id = np.argmax(pred)
            
            true_emotion = EMOTION_LABELS[true_emotion_id]
            pred_emotion = EMOTION_LABELS[pred_emotion_id]
            confidence = pred[pred_emotion_id] * 100
            
            print(f"   Sample {i+1}: True={true_emotion} | Pred={pred_emotion} ({confidence:.1f}%)")
            print(f"             Spread: {pred.std():.3f}")
    except Exception as e:
        print(f"⚠️ Testing error: {e}")
    
    return model, history

if __name__ == "__main__":
    # Train improved model
    data_path = "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013/train"
    
    print("🎯 IMPROVED MULTI-EMOTION TRAINING (FIXED)")
    print("="*50)
    print("✅ Balanced dataset")
    print("✅ Better architecture")  
    print("✅ Data augmentation")
    print("✅ Improved callbacks")
    print("✅ Fixed top_2_accuracy error")
    print("="*50)
    
    try:
        model, history = train_improved_multi_emotion_model(data_path)
        
        if model is not None:
            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Check model files: multi_emotion_model_IMPROVED.h5")
            print(f"   2. Test with: python app_multi_emotion.py")
            print(f"   3. Look for better prediction variation!")
        else:
            print(f"\n❌ TRAINING FAILED!")
            print(f"💡 Check dataset path and permissions")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"💡 Please check:")
        print(f"   1. Dataset path exists")
        print(f"   2. Emotion folders are present")
        print(f"   3. Images are readable")