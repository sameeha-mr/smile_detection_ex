import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import random

# NEW emotion mapping (what your new model was trained on)
NEW_TRAINING_EMOTION_TO_HAPPINESS = {
    0: 35,   # Angry -> 35% (UPDATED from 12%)
    1: 30,   # Disgust -> 30% (UPDATED from 8%)
    2: 35,   # Fear -> 35% (UPDATED from 18%)
    3: 65,   # Happy -> 65% (UPDATED from 88%)
    4: 30,   # Sad -> 30% (UPDATED from 10%)
    5: 60,   # Surprise -> 60% (UPDATED from 75%)
    6: 50    # Neutral -> 50% (unchanged)
}

# OLD emotion mapping (for comparison)
OLD_TRAINING_EMOTION_TO_HAPPINESS = {
    0: 12,   # Angry -> 12%
    1: 8,    # Disgust -> 8%
    2: 18,   # Fear -> 18%
    3: 88,   # Happy -> 88%
    4: 10,   # Sad -> 10%
    5: 75,   # Surprise -> 75%
    6: 50    # Neutral -> 50%
}

EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

def load_test_model():
    """Load the NEW model for evaluation - UPDATED MODEL PATHS"""
    new_model_paths = [
        "happiness_model_memory_efficient.h5",             # NEW original
        "happiness_model_memory_final.h5",                 # NEW original
    ]
    
    for model_path in new_model_paths:
        if os.path.exists(model_path):
            try:
                print(f"📁 Loading NEW model: {model_path}")
                
                # Try loading without compilation first (safer)
                model = load_model(model_path, compile=False)
                
                # Recompile with simple settings
                model.compile(
                    optimizer='adam', 
                    loss='mse', 
                    metrics=['mae']
                )
                
                print(f"✅ NEW model loaded successfully!")
                
                # Test prediction to verify model works
                test_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
                test_pred = model.predict(test_input, verbose=0)
                print(f"🧪 Test prediction: {test_pred[0][0]:.3f} ({test_pred[0][0]*100:.1f}%)")
                
                return model, model_path
                
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue
    
    print("❌ No NEW model could be loaded!")
    return None, None

def preprocess_image_new(image_path):
    """Preprocess image exactly like NEW training (RGB, 64x64)"""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert BGR to RGB (IMPORTANT for new model)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 64x64 (same as training)
        img = cv2.resize(img, (64, 64))
        
        # Normalize to 0-1 (same as training)
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def collect_test_samples_new(data_path, max_per_emotion=100):
    """Collect test samples using NEW emotion mapping"""
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        print("💡 Available test paths to try:")
        possible_paths = [
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013/test",
            "E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/fer2013",
            "fer2013",
            "FER2013",
            "data/fer2013"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ Found: {path}")
            else:
                print(f"   ❌ Not found: {path}")
        return None, None, None
    
    print(f"🔍 Collecting test samples from: {data_path}")
    
    test_images = []
    true_happiness_new = []  # Using NEW mapping
    true_happiness_old = []  # Using OLD mapping (for comparison)
    emotion_counts = {}
    
    # Look for emotion folders
    for emotion_id, emotion_name in EMOTION_LABELS.items():
        emotion_folder = os.path.join(data_path, emotion_name)
        if not os.path.exists(emotion_folder):
            print(f"⚠️ Emotion folder not found: {emotion_folder}")
            continue
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(emotion_folder).glob(ext))
            image_files.extend(Path(emotion_folder).glob(ext.upper()))
        
        # Limit samples per emotion
        if len(image_files) > max_per_emotion:
            image_files = random.sample(image_files, max_per_emotion)
        
        emotion_counts[emotion_name] = len(image_files)
        print(f"   {emotion_name}: {len(image_files)} images (NEW: {NEW_TRAINING_EMOTION_TO_HAPPINESS[emotion_id]}% | OLD: {OLD_TRAINING_EMOTION_TO_HAPPINESS[emotion_id]}%)")
        
        # Process images
        for img_path in image_files:
            processed_img = preprocess_image_new(img_path)
            if processed_img is not None:
                test_images.append(processed_img[0])  # Remove batch dimension
                # NEW mapping labels (what the model should predict)
                true_happiness_new.append(NEW_TRAINING_EMOTION_TO_HAPPINESS[emotion_id] / 100.0)
                # OLD mapping labels (for comparison)
                true_happiness_old.append(OLD_TRAINING_EMOTION_TO_HAPPINESS[emotion_id] / 100.0)
    
    print(f"✅ Collected {len(test_images)} test samples")
    print("   Emotion distribution:", emotion_counts)
    
    return np.array(test_images), np.array(true_happiness_new), np.array(true_happiness_old)

def evaluate_new_model_accuracy(model, test_images, true_happiness_new, true_happiness_old, model_path):
    """Evaluate NEW model accuracy with both mappings"""
    print("\n🧪 EVALUATING NEW MODEL ACCURACY")
    print("="*60)
    print(f"📁 Model: {model_path}")
    
    # Get predictions
    print("Getting model predictions...")
    predictions = model.predict(test_images, verbose=1)
    pred_happiness = predictions.flatten()  # Convert to 1D
    
    print(f"\n📊 PREDICTION ANALYSIS:")
    print(f"   Prediction range: {pred_happiness.min()*100:.1f}% - {pred_happiness.max()*100:.1f}%")
    print(f"   Prediction mean: {pred_happiness.mean()*100:.1f}%")
    print(f"   Prediction std: {pred_happiness.std()*100:.1f}%")
    
    # Evaluate against NEW mapping (correct)
    print(f"\n🆕 ACCURACY vs NEW MAPPING (Correct):")
    mse_new = mean_squared_error(true_happiness_new, pred_happiness)
    rmse_new = np.sqrt(mse_new)
    mae_new = mean_absolute_error(true_happiness_new, pred_happiness)
    r2_new = r2_score(true_happiness_new, pred_happiness)
    
    # Convert to percentage
    true_percent_new = true_happiness_new * 100
    pred_percent = pred_happiness * 100
    mae_percent_new = mean_absolute_error(true_percent_new, pred_percent)
    rmse_percent_new = np.sqrt(mean_squared_error(true_percent_new, pred_percent))
    
    print(f"   Mean Absolute Error: {mae_new:.4f} ({mae_percent_new:.1f}% scale)")
    print(f"   Root Mean Squared Error: {rmse_new:.4f} ({rmse_percent_new:.1f}% scale)")
    print(f"   R² Score: {r2_new:.4f}")
    
    # Accuracy level for NEW mapping
    if mae_percent_new < 8:
        accuracy_level_new = "EXCELLENT"
    elif mae_percent_new < 12:
        accuracy_level_new = "VERY GOOD"
    elif mae_percent_new < 18:
        accuracy_level_new = "GOOD"
    elif mae_percent_new < 25:
        accuracy_level_new = "FAIR"
    elif mae_percent_new < 35:
        accuracy_level_new = "POOR"
    else:
        accuracy_level_new = "VERY POOR"
    
    print(f"   📈 NEW Mapping Accuracy: {accuracy_level_new}")
    print(f"   💡 Average error: ±{mae_percent_new:.1f}% happiness")
    
    # Evaluate against OLD mapping (for comparison)
    print(f"\n📊 ACCURACY vs OLD MAPPING (Comparison):")
    mse_old = mean_squared_error(true_happiness_old, pred_happiness)
    mae_old = mean_absolute_error(true_happiness_old, pred_happiness)
    r2_old = r2_score(true_happiness_old, pred_happiness)
    
    true_percent_old = true_happiness_old * 100
    mae_percent_old = mean_absolute_error(true_percent_old, pred_percent)
    
    print(f"   Mean Absolute Error: {mae_old:.4f} ({mae_percent_old:.1f}% scale)")
    print(f"   R² Score: {r2_old:.4f}")
    print(f"   💡 Average error: ±{mae_percent_old:.1f}% happiness")
    
    # Compare mappings
    print(f"\n🔄 MAPPING COMPARISON:")
    if mae_percent_new < mae_percent_old:
        improvement = mae_percent_old - mae_percent_new
        print(f"   ✅ NEW mapping is BETTER by {improvement:.1f}% error reduction!")
        print(f"   📈 Improvement: {improvement/mae_percent_old*100:.1f}%")
    else:
        degradation = mae_percent_new - mae_percent_old
        print(f"   ❌ NEW mapping is WORSE by {degradation:.1f}% error increase")
    
    # Show sample predictions with both mappings
    print(f"\n📋 SAMPLE PREDICTIONS (NEW vs OLD mapping):")
    indices = np.random.choice(len(true_happiness_new), min(8, len(true_happiness_new)), replace=False)
    for i, idx in enumerate(indices):
        true_new = true_percent_new[idx]
        true_old = true_percent_old[idx]
        pred_val = pred_percent[idx]
        error_new = abs(true_new - pred_val)
        error_old = abs(true_old - pred_val)
        
        # Find which emotion this is
        emotion_name = "unknown"
        for emo_id, emo_label in EMOTION_LABELS.items():
            if abs(true_new - NEW_TRAINING_EMOTION_TO_HAPPINESS[emo_id]) < 1:
                emotion_name = emo_label
                break
        
        print(f"   {emotion_name.upper()}: Pred={pred_val:.1f}% | NEW_True={true_new:.1f}% (±{error_new:.1f}%) | OLD_True={true_old:.1f}% (±{error_old:.1f}%)")
    
    return {
        'new_mapping': {
            'mse': mse_new, 'rmse': rmse_new, 'mae': mae_new, 'r2': r2_new,
            'mae_percent': mae_percent_new, 'accuracy_level': accuracy_level_new
        },
        'old_mapping': {
            'mse': mse_old, 'mae': mae_old, 'r2': r2_old, 'mae_percent': mae_percent_old
        },
        'predictions': pred_percent
    }

def analyze_new_prediction_distribution(model, test_images):
    """Analyze NEW model prediction distribution"""
    print("\n📊 NEW MODEL PREDICTION DISTRIBUTION")
    print("="*50)
    
    predictions = model.predict(test_images, verbose=0)
    pred_happiness = predictions.flatten() * 100  # Convert to percentage
    
    print(f"📈 Prediction Statistics:")
    print(f"   Min prediction: {pred_happiness.min():.1f}%")
    print(f"   Max prediction: {pred_happiness.max():.1f}%")
    print(f"   Mean prediction: {pred_happiness.mean():.1f}%")
    print(f"   Median prediction: {np.median(pred_happiness):.1f}%")
    print(f"   Std deviation: {pred_happiness.std():.1f}%")
    
    # Check ranges based on NEW mapping
    very_low = np.sum(pred_happiness < 25)    # Below sad/angry range
    low = np.sum((pred_happiness >= 25) & (pred_happiness < 40))        # Sad/angry range
    middle = np.sum((pred_happiness >= 40) & (pred_happiness < 55))     # Neutral range
    high = np.sum((pred_happiness >= 55) & (pred_happiness < 70))       # Happy/surprise range
    very_high = np.sum(pred_happiness >= 70)  # Above happy range
    
    total = len(pred_happiness)
    print(f"\n📊 NEW Mapping Prediction Ranges:")
    print(f"   Very low (<25%): {very_low} samples ({very_low/total*100:.1f}%)")
    print(f"   Low (25-40%): {low} samples ({low/total*100:.1f}%)")
    print(f"   Middle (40-55%): {middle} samples ({middle/total*100:.1f}%)")
    print(f"   High (55-70%): {high} samples ({high/total*100:.1f}%)")
    print(f"   Very high (>70%): {very_high} samples ({very_high/total*100:.1f}%)")
    
    # Health check for NEW model
    if pred_happiness.std() < 8:
        print("⚠️ WARNING: Very low variation - model might be stuck!")
    elif pred_happiness.std() > 25:
        print("⚠️ WARNING: Very high variation - model might be unstable!")
    else:
        print("✅ Good prediction variation for NEW mapping!")
    
    if very_low + very_high > total * 0.3:
        print("⚠️ WARNING: Too many extreme predictions!")
    else:
        print("✅ Reasonable distribution of predictions!")
    
    return pred_happiness

def plot_new_model_comparison(true_happiness_new, true_happiness_old, pred_happiness, model_path):
    """Plot NEW model results with comparison"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Predictions vs NEW mapping (correct)
        plt.subplot(2, 3, 1)
        plt.scatter(true_happiness_new * 100, pred_happiness, alpha=0.6, color='blue')
        plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        plt.xlabel('True Happiness - NEW Mapping (%)')
        plt.ylabel('Predicted Happiness (%)')
        plt.title('NEW Model vs NEW Mapping (Correct)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Predictions vs OLD mapping (comparison)
        plt.subplot(2, 3, 2)
        plt.scatter(true_happiness_old * 100, pred_happiness, alpha=0.6, color='orange')
        plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
        plt.xlabel('True Happiness - OLD Mapping (%)')
        plt.ylabel('Predicted Happiness (%)')
        plt.title('NEW Model vs OLD Mapping (Comparison)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution for NEW mapping
        plt.subplot(2, 3, 3)
        errors_new = pred_happiness - (true_happiness_new * 100)
        plt.hist(errors_new, bins=20, alpha=0.7, edgecolor='black', color='blue')
        plt.xlabel('Prediction Error (%)')
        plt.ylabel('Frequency')
        plt.title('NEW Mapping Error Distribution')
        plt.axvline(0, color='red', linestyle='--', label='Perfect')
        plt.axvline(errors_new.mean(), color='green', linestyle='-', label=f'Mean: {errors_new.mean():.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution for OLD mapping
        plt.subplot(2, 3, 4)
        errors_old = pred_happiness - (true_happiness_old * 100)
        plt.hist(errors_old, bins=20, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Prediction Error (%)')
        plt.ylabel('Frequency')
        plt.title('OLD Mapping Error Distribution')
        plt.axvline(0, color='red', linestyle='--', label='Perfect')
        plt.axvline(errors_old.mean(), color='green', linestyle='-', label=f'Mean: {errors_old.mean():.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Prediction distribution
        plt.subplot(2, 3, 5)
        plt.hist(pred_happiness, bins=20, alpha=0.7, edgecolor='black', color='purple')
        plt.xlabel('Predicted Happiness (%)')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution')
        plt.axvline(pred_happiness.mean(), color='red', linestyle='-', label=f'Mean: {pred_happiness.mean():.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Mapping comparison
        plt.subplot(2, 3, 6)
        emotions = list(EMOTION_LABELS.values())
        new_values = [NEW_TRAINING_EMOTION_TO_HAPPINESS[i] for i in range(7)]
        old_values = [OLD_TRAINING_EMOTION_TO_HAPPINESS[i] for i in range(7)]
        
        x = np.arange(len(emotions))
        width = 0.35
        
        plt.bar(x - width/2, new_values, width, label='NEW Mapping', alpha=0.8, color='blue')
        plt.bar(x + width/2, old_values, width, label='OLD Mapping', alpha=0.8, color='orange')
        
        plt.xlabel('Emotions')
        plt.ylabel('Happiness (%)')
        plt.title('Emotion-to-Happiness Mapping Comparison')
        plt.xticks(x, emotions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'NEW Model Evaluation: {os.path.basename(model_path)}', fontsize=16)
        plt.tight_layout()
        
        # Save plots
        plot_filename = f'new_model_evaluation_{os.path.basename(model_path).replace(".h5", "")}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📊 NEW model evaluation plots saved as: {plot_filename}")
        plt.show()
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def main():
    """Main evaluation function for NEW models"""
    print("🔍 NEW HAPPINESS MODEL ACCURACY EVALUATION")
    print("="*70)
    print("🆕 Testing models with UPDATED emotion-to-happiness mapping")
    print("="*70)
    
    # Load NEW model
    model, model_path = load_test_model()
    if model is None:
        return
    
    # Collect test data with both mappings
    test_images, true_happiness_new, true_happiness_old = collect_test_samples_new(
        data_path="E:/Academic/Y4S1/CMIS 4+26 Research Project/smile_detection_ex/FER/fer2013/test",
        max_per_emotion=50  # Reduced for faster testing
    )
    
    if test_images is None:
        print("❌ Could not collect test data")
        print("💡 Please update the data_path in the script")
        return
    
    # Evaluate accuracy with both mappings
    metrics = evaluate_new_model_accuracy(model, test_images, true_happiness_new, true_happiness_old, model_path)
    
    # Analyze prediction distribution
    pred_happiness = analyze_new_prediction_distribution(model, test_images)
    
    # Create comparison plots
    plot_new_model_comparison(true_happiness_new, true_happiness_old, pred_happiness, model_path)
    
    # Final recommendations for NEW model
    print(f"\n💡 NEW MODEL RECOMMENDATIONS:")
    new_accuracy = metrics['new_mapping']['accuracy_level']
    new_mae = metrics['new_mapping']['mae_percent']
    
    if new_mae < 15:
        print(f"✅ NEW model accuracy is GOOD!")
        print(f"🎯 Average error: ±{new_mae:.1f}% - This is acceptable for happiness prediction")
        print(f"💡 The updated emotion-to-happiness mapping is working well!")
    elif new_mae < 25:
        print(f"⚠️ NEW model accuracy is FAIR")
        print(f"🔧 Could be improved with:")
        print(f"   - More training epochs")
        print(f"   - Data augmentation")
        print(f"   - Fine-tuning the mapping further")
    else:
        print(f"❌ NEW model accuracy needs improvement!")
        print(f"🔧 Suggested fixes:")
        print(f"   - Check if model was trained with correct mapping")
        print(f"   - Consider retraining with more balanced data")
        print(f"   - Verify preprocessing pipeline")
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   NEW Mapping Error: ±{metrics['new_mapping']['mae_percent']:.1f}%")
    print(f"   NEW Mapping Accuracy: {metrics['new_mapping']['accuracy_level']}")
    print(f"   NEW Mapping R²: {metrics['new_mapping']['r2']:.3f}")
    print(f"   OLD Mapping Error: ±{metrics['old_mapping']['mae_percent']:.1f}%")
    
    if metrics['new_mapping']['mae_percent'] < metrics['old_mapping']['mae_percent']:
        improvement = metrics['old_mapping']['mae_percent'] - metrics['new_mapping']['mae_percent']
        print(f"   🎉 Improvement: {improvement:.1f}% better accuracy with NEW mapping!")
    
    print("="*70)

if __name__ == "__main__":
    main()