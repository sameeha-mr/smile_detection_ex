from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam  # MOVED UP
from tensorflow.keras import metrics, losses   # ADDED: Import at the top!
import logging
import mediapipe as mp

# Import fuzzy logic and feature extraction modules
try:
    from Backup.fuzzy import compute_happiness
    from Backup.extract_features import extract_features
    MODULES_AVAILABLE = True
    print("✅ Custom modules loaded successfully")
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"❌ Error importing custom modules: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variables
happiness_cnn_model = None
legacy_smile_model = None
optimal_threshold = 0.52

# FER emotion to happiness mapping (same as training)
EMOTION_TO_HAPPINESS = {
  0: 35,   # Angry -> 35% (was 25%)
    1: 30,   # Disgust -> 30% (was 20%)
    2: 35,   # Fear -> 35% (was 30%)
    3: 65,   # Happy -> 65% (was 70%)
    4: 30,   # Sad -> 30% (was 22%)
    5: 60,   # Surprise -> 60% (was 65%)
    6: 50    # Neutral -> 50% (unchanged)
    }

def load_happiness_model():
    """Load the FIXED FER happiness model - IMPORTS FIXED"""
    global happiness_cnn_model
    
    try:
        # Try loading the fixed model first
        model_paths = [
           # "happiness_model_memory_efficient_FIXED.h5",  # Fixed version
            "happiness_model_memory_final_FIXED.h5",      # Fixed version
           "happiness_model_memory_efficient.h5",        # Original (will need fixing)
            "happiness_model_memory_final.h5"             # Original (will need fixing)
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                logger.info(f"🔍 Trying to load: {model_path}")
                
                # Try Method 1: Load without compilation and recompile (SIMPLEST)
                try:
                    logger.info("🔄 Loading without compilation...")
                    happiness_cnn_model = load_model(model_path, compile=False)
                    
                    # Recompile manually with simple strings (SAFEST)
                    happiness_cnn_model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='mse',  # Simple string - no imports needed
                        metrics=['mae']  # Simple string - no imports needed
                    )
                    logger.info("✅ Model loaded and recompiled successfully!")
                    
                except Exception as e1:
                    logger.warning(f"Simple recompile failed: {e1}")
                    
                    # Try Method 2: Load with custom objects (NOW IMPORTS ARE AVAILABLE)
                    try:
                        logger.info("🔄 Trying with custom objects...")
                        
                        # NOW this will work because metrics is imported at the top
                        custom_objects = {
                            'mse': metrics.mean_squared_error,
                            'mae': metrics.mean_absolute_error,
                            'mean_squared_error': metrics.mean_squared_error,
                            'mean_absolute_error': metrics.mean_absolute_error
                        }
                        
                        happiness_cnn_model = load_model(model_path, custom_objects=custom_objects)
                        logger.info("✅ Model loaded with custom objects!")
                        
                    except Exception as e2:
                        logger.error(f"Both methods failed for {model_path}: {e1}, {e2}")
                        continue
                
                # Test the model
                try:
                    test_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
                    test_prediction = happiness_cnn_model.predict(test_input, verbose=0)
                    logger.info(f"🧪 Test prediction: {test_prediction[0][0]:.3f}")
                    logger.info(f"✅ FER happiness model loaded from: {model_path}")
                    return True
                except Exception as test_error:
                    logger.error(f"Model test failed: {test_error}")
                    happiness_cnn_model = None
                    continue
        
        logger.error("❌ No compatible model found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error loading happiness model: {e}")
        return False

def load_legacy_smile_model():
    """DISABLED: Legacy model loading disabled for testing"""
    logger.info("🚫 Legacy model loading disabled for FER+Fuzzy testing")
    return False

def preprocess_image_for_happiness_cnn(image_array):
    """Preprocess image for FER-2013 happiness model (RGB input)"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3:
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            # Convert grayscale to RGB
            img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Resize to 64x64 (same as training)
        img = cv2.resize(img, (64, 64))
        
        # Normalize to 0-1 range (same as training)
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        logger.info(f"Happiness CNN input shape: {img.shape}")
        
        return img
        
    except Exception as e:
        logger.error(f"Error in happiness CNN preprocessing: {e}")
        raise

def preprocess_image_for_legacy_cnn(image_array):
    """Preprocess image for legacy smile model (grayscale input)"""
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            img = image_array
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Resize and normalize
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        
        # Keep as grayscale (1 channel)
        img = img.reshape(1, 64, 64, 1).astype(np.float32)
        
        logger.info(f"Legacy CNN input shape: {img.shape}")
        
        return img
        
    except Exception as e:
        logger.error(f"Error in legacy CNN preprocessing: {e}")
        raise

def get_happiness_from_fer_cnn(image_array):
    """Get happiness score from FER-2013 trained model (0-100%)"""
    global happiness_cnn_model
    
    if happiness_cnn_model is None:
        raise ValueError("Happiness CNN model not loaded")
    
    try:
        # Preprocess for FER happiness model
        img_processed = preprocess_image_for_happiness_cnn(image_array)
        
        # Predict happiness (model outputs 0-1, convert to 0-100%)
        happiness_score = happiness_cnn_model.predict(img_processed, verbose=0)[0][0] * 100
        
        logger.info(f"FER CNN Happiness Score: {happiness_score:.1f}%")
        return float(happiness_score)
        
    except Exception as e:
        logger.error(f"Error in FER CNN happiness prediction: {e}")
        raise

def get_legacy_smile_score(image_array):
    """Get smile score from legacy model (fallback)"""
    global legacy_smile_model
    
    if legacy_smile_model is None:
        return 0.5  # Default neutral
    
    try:
        # Preprocess for legacy smile model
        img_processed = preprocess_image_for_legacy_cnn(image_array)
        
        # Predict smile (0-1)
        smile_score = legacy_smile_model.predict(img_processed, verbose=0)[0][0]
        
        logger.info(f"Legacy Smile Score: {smile_score:.3f}")
        return float(smile_score)
        
    except Exception as e:
        logger.error(f"Error in legacy smile prediction: {e}")
        return 0.5

def classify_happiness_level(happiness_score):
    """Classify happiness score into levels with emojis"""
    if happiness_score >= 80:
        return "Very Happy", "😄", "You look extremely joyful!"
    elif happiness_score >= 65:
        return "Happy", "😊", "You appear quite happy!"
    elif happiness_score >= 45:
        return "Neutral", "😐", "You seem calm and neutral."
    elif happiness_score >= 25:
        return "Slightly Sad", "😔", "You appear a bit down."
    elif happiness_score >= 15:
        return "Sad", "😢", "You look quite sad."
    else:
        return "Very Sad", "😭", "You appear very unhappy."

def get_enhanced_happiness_score(image_array):
    """
    Get happiness score using ONLY FER CNN (NO FUZZY, NO LEGACY)
    """
    global happiness_cnn_model
    
    try:
        # Method 1: FER-2013 happiness prediction (ONLY METHOD)
        if happiness_cnn_model is not None:
            fer_happiness = get_happiness_from_fer_cnn(image_array)
            logger.info(f"🎯 FER CNN Happiness: {fer_happiness:.1f}%")
            
            # Use FER score directly as final result
            final_happiness = fer_happiness
            method = "fer_only"
            
        else:
            # Fallback if FER model not available
            final_happiness = 50.0
            method = "neutral_fallback"
            logger.warning("❌ FER happiness model not available")
        
        # Ensure score is within bounds
        final_happiness = max(0, min(100, final_happiness))
        
        # Classify happiness level
        happiness_level, emoji, description = classify_happiness_level(final_happiness)
        
        logger.info(f"🎯 FER ONLY RESULT: {final_happiness:.1f}% - {happiness_level} {emoji}")
        
        return {
            'happiness_score': float(final_happiness),
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'fer_cnn_score': float(final_happiness),
            'method': method,
            'model_available': happiness_cnn_model is not None,
            'fer_only_mode': True
        }
        
    except Exception as e:
        logger.error(f"Error in FER-only happiness calculation: {e}")
        # Return fallback result
        happiness_level, emoji, description = classify_happiness_level(50.0)
        return {
            'happiness_score': 50.0,
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'fer_cnn_score': 50.0,
            'method': 'error_fallback',
            'model_available': False,
            'fer_only_mode': True
        }

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict happiness score using ONLY FER-2013 CNN
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image data
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get FER happiness analysis ONLY
        result = get_enhanced_happiness_score(image)
        
        # Extract results
        happiness_score = result['happiness_score']
        happiness_level = result['happiness_level']
        emoji = result['emoji']
        description = result['description']
        fer_score = result['fer_cnn_score']
        
        # DEBUG: Log the FER values
        logger.info(f"🔍 DEBUG - FER Score: {fer_score:.1f}%")
        logger.info(f"🔍 DEBUG - Final Happiness: {happiness_score:.1f}%")
        logger.info(f"🔍 DEBUG - Classification: {happiness_level} {emoji}")
        logger.info(f"🔍 DEBUG - Method: {result['method']}")
        
        # Prepare clean FER-only response
        response = {
            'happiness_score': happiness_score,
            'happiness_percentage': happiness_score,
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'fer_cnn_score': fer_score,
            'fer_percentage': fer_score,
            'is_happy': bool(happiness_score > 50),
            'confidence': float(max(happiness_score, 100 - happiness_score)),
            'method': result['method'],
            'model_available': result['model_available'],
            'fer_only_mode': True,
            'fuzzy_disabled': True,
            'legacy_disabled': True
        }
        
        logger.info(f"🎯 FER-ONLY PREDICTION: {happiness_score:.1f}% ({happiness_level}) {emoji}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in FER-only prediction: {e}")
        return jsonify({'error': f'FER prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - FER only"""
    global happiness_cnn_model
    
    return jsonify({
        'status': 'healthy',
        'fer_model_loaded': happiness_cnn_model is not None,
        'legacy_model_loaded': False,
        'modules_available': False,  # Disabled
        'method': 'fer_only',
        'fuzzy_disabled': True,
        'legacy_disabled': True
    })

@app.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get available prediction capabilities - FER only"""
    global happiness_cnn_model
    
    return jsonify({
        'fer_happiness_prediction': happiness_cnn_model is not None,
        'legacy_smile_detection': False,
        'feature_extraction': False,
        'fuzzy_logic': False,
        'method': 'fer_only',
        'emotion_mapping': EMOTION_TO_HAPPINESS,
        'fuzzy_disabled': True,
        'legacy_disabled': True
    })

@app.route('/models', methods=['GET'])
def get_model_info():
    """Get detailed model information - FER only"""
    global happiness_cnn_model
    
    # Check for FER model files only
    fer_model_files = [
        'happiness_model_memory_efficient.h5',
        'happiness_model_memory_final.h5',
        'happiness_model_fer_folder.h5',
        'happiness_model_final.h5'
    ]
    
    file_status = {}
    for model_file in fer_model_files:
        file_status[model_file] = os.path.exists(model_file)
    
    return jsonify({
        'loaded_models': {
            'fer_happiness_model': happiness_cnn_model is not None,
            'legacy_smile_model': False,
            'fuzzy_logic': False
        },
        'model_files': file_status,
        'emotion_mapping': EMOTION_TO_HAPPINESS,
        'input_size': '64x64x3 (RGB)',
        'output_range': '0-100% happiness',
        'mode': 'fer_only'
    })

# Disable fuzzy and legacy completely
def load_legacy_smile_model():
    """DISABLED: Legacy model completely disabled for FER-only testing"""
    logger.info("🚫 Legacy model loading disabled for FER-only testing")
    return False

if __name__ == '__main__':
    print("🚀 Starting FER-ONLY Happiness Detection Server...")
    print("="*70)
    print("🧠 Method: FER-2013 CNN ONLY (Pure Emotion-Based)")
    print("🚫 Fuzzy Logic: DISABLED")
    print("🚫 Legacy Smile: DISABLED") 
    print("="*70)
    
    # Load only FER model
    fer_loaded = load_happiness_model()
    
    if fer_loaded:
        print(f"✅ FER happiness model loaded successfully!")
        print(f"🎯 Pure emotion-based happiness prediction ready")
        print(f"🧠 Using FER-2013 emotion → happiness mapping")
    else:
        print(f"❌ FER happiness model not found")
        print(f"⚠️  Server will return neutral scores as fallback")
    
    print(f"🚫 Fuzzy logic: DISABLED for clean testing")
    print(f"🚫 Legacy smile: DISABLED for clean testing")
    print(f"🌐 Server will start on http://localhost:5000")
    print(f"📡 Available endpoints:")
    print(f"   - POST /predict (FER-2013 ONLY)")
    print(f"   - GET /health")
    print(f"   - GET /capabilities")
    print(f"   - GET /models")
    print("="*70)
    
    if not fer_loaded:
        print("❌ No FER model available!")
        print("💡 Train FER model: python train_fer_happiness.py")
        print("🔄 Continuing with fallback mode...")
    
    # Start the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )