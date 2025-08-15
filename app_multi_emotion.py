from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variable
multi_emotion_model = None

# Emotion labels
EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

# ADJUSTABLE happiness weights (no retraining needed!)
EMOTION_TO_HAPPINESS_WEIGHTS = {
    0: 0.20,  # Angry → 20% happiness contribution
    1: 0.15,  # Disgust → 15% happiness contribution  
    2: 0.25,  # Fear → 25% happiness contribution
    3: 0.80,  # Happy → 80% happiness contribution
    4: 0.18,  # Sad → 18% happiness contribution
    5: 0.75,  # Surprise → 75% happiness contribution
    6: 0.50   # Neutral → 50% happiness contribution
}

def load_multi_emotion_model():
    """Load the multi-emotion classification model"""
    global multi_emotion_model
    
    model_paths = [
       # "multi_emotion_model_best.h5",
        #"multi_emotion_model_final.h5",
        "multi_emotion_model_IMPROVED_final.h5"    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                multi_emotion_model = load_model(model_path)
                logger.info(f"✅ Multi-emotion model loaded: {model_path}")
                
                # Test prediction
                test_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
                test_pred = multi_emotion_model.predict(test_input, verbose=0)
                logger.info(f"🧪 Test prediction shape: {test_pred.shape}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
                continue
    
    logger.error("❌ No multi-emotion model found!")
    return False

def preprocess_image_for_emotions(image_array):
    """Preprocess image for emotion classification"""
    try:
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3:
            img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Resize to 64x64
        img = cv2.resize(img, (64, 64))
        
        # Normalize to 0-1
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img
        
    except Exception as e:
        logger.error(f"Error in emotion preprocessing: {e}")
        raise

def get_emotion_probabilities(image_array):
    """Get emotion probabilities from the multi-emotion model"""
    global multi_emotion_model
    
    if multi_emotion_model is None:
        raise ValueError("Multi-emotion model not loaded")
    
    try:
        # Preprocess image
        img_processed = preprocess_image_for_emotions(image_array)
        
        # Get emotion probabilities
        emotion_probs = multi_emotion_model.predict(img_processed, verbose=0)[0]
        
        logger.info(f"🎭 Emotion probabilities: {emotion_probs}")
        return emotion_probs
        
    except Exception as e:
        logger.error(f"Error in emotion prediction: {e}")
        raise

def calculate_happiness_from_emotions(emotion_probabilities):
    """Calculate happiness score from emotion probabilities"""
    happiness_score = 0.0
    contributions = {}
    
    for emotion_id, probability in enumerate(emotion_probabilities):
        weight = EMOTION_TO_HAPPINESS_WEIGHTS[emotion_id]
        contribution = probability * weight
        happiness_score += contribution
        
        emotion_name = EMOTION_LABELS[emotion_id]
        contributions[emotion_name] = {
            'probability': float(probability),
            'weight': weight,
            'contribution': float(contribution)
        }
    
    # Convert to percentage
    happiness_percentage = happiness_score * 100
    
    return happiness_percentage, contributions

def classify_happiness_level(happiness_score):
    """Classify happiness score into levels"""
    if happiness_score >= 70:
        return "Very Happy", "😄", "You look very joyful!"
    elif happiness_score >= 55:
        return "Happy", "😊", "You appear quite happy!"
    elif happiness_score >= 45:
        return "Neutral", "😐", "You seem calm and neutral."
    elif happiness_score >= 30:
        return "Slightly Sad", "😔", "You appear a bit down."
    elif happiness_score >= 20:
        return "Sad", "😢", "You look quite sad."
    else:
        return "Very Sad", "😭", "You appear very unhappy."

def get_enhanced_emotion_analysis(image_array):
    """Get comprehensive emotion analysis with happiness calculation"""
    global multi_emotion_model
    
    try:
        if multi_emotion_model is None:
            raise ValueError("Multi-emotion model not available")
        
        # Get emotion probabilities
        emotion_probs = get_emotion_probabilities(image_array)
        
        # Calculate happiness from emotions
        happiness_score, contributions = calculate_happiness_from_emotions(emotion_probs)
        
        # Get dominant emotion
        dominant_emotion_id = np.argmax(emotion_probs)
        dominant_emotion_name = EMOTION_LABELS[dominant_emotion_id]
        dominant_emotion_confidence = emotion_probs[dominant_emotion_id] * 100
        
        # Classify happiness level
        happiness_level, emoji, description = classify_happiness_level(happiness_score)
        
        # Prepare emotion breakdown
        emotion_breakdown = {}
        for emotion_id, prob in enumerate(emotion_probs):
            emotion_name = EMOTION_LABELS[emotion_id]
            emotion_breakdown[emotion_name] = {
                'probability': float(prob),
                'percentage': float(prob * 100)
            }
        
        logger.info(f"🎭 Dominant: {dominant_emotion_name} ({dominant_emotion_confidence:.1f}%)")
        logger.info(f"🎯 Happiness: {happiness_score:.1f}% - {happiness_level}")
        
        return {
            'happiness_score': float(happiness_score),
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'dominant_emotion': {
                'name': dominant_emotion_name,
                'confidence': float(dominant_emotion_confidence)
            },
            'emotion_breakdown': emotion_breakdown,
            'happiness_contributions': contributions,
            'method': 'multi_emotion_classification',
            'model_available': True
        }
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        # Return fallback
        happiness_level, emoji, description = classify_happiness_level(50.0)
        return {
            'happiness_score': 50.0,
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'dominant_emotion': {'name': 'unknown', 'confidence': 0.0},
            'emotion_breakdown': {},
            'happiness_contributions': {},
            'method': 'error_fallback',
            'model_available': False
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotions and happiness using multi-emotion model"""
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
        
        # Get comprehensive emotion analysis
        result = get_enhanced_emotion_analysis(image)
        
        # Extract key results
        happiness_score = result['happiness_score']
        happiness_level = result['happiness_level']
        emoji = result['emoji']
        description = result['description']
        dominant_emotion = result['dominant_emotion']
        emotion_breakdown = result['emotion_breakdown']
        
        # Prepare response
        response = {
            'happiness_score': happiness_score,
            'happiness_percentage': happiness_score,
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description,
            'dominant_emotion': dominant_emotion,
            'all_emotions': emotion_breakdown,
            'happiness_breakdown': result['happiness_contributions'],
            'is_happy': bool(happiness_score > 50),
            'confidence': float(dominant_emotion['confidence']),
            'method': result['method'],
            'model_available': result['model_available'],
            'weights_used': EMOTION_TO_HAPPINESS_WEIGHTS
        }
        
        logger.info(f"🎯 MULTI-EMOTION PREDICTION:")
        logger.info(f"   Dominant: {dominant_emotion['name']} ({dominant_emotion['confidence']:.1f}%)")
        logger.info(f"   Happiness: {happiness_score:.1f}% ({happiness_level}) {emoji}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in multi-emotion prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/adjust_weights', methods=['POST'])
def adjust_happiness_weights():
    """Adjust happiness calculation weights without retraining!"""
    global EMOTION_TO_HAPPINESS_WEIGHTS
    
    try:
        new_weights = request.json.get('weights', {})
        
        # Update weights
        for emotion_name, weight in new_weights.items():
            # Convert emotion name to ID
            emotion_id = None
            for eid, ename in EMOTION_LABELS.items():
                if ename.lower() == emotion_name.lower():
                    emotion_id = eid
                    break
            
            if emotion_id is not None:
                EMOTION_TO_HAPPINESS_WEIGHTS[emotion_id] = float(weight)
                logger.info(f"Updated {emotion_name} weight to {weight}")
        
        return jsonify({
            'success': True,
            'updated_weights': EMOTION_TO_HAPPINESS_WEIGHTS,
            'message': 'Happiness weights updated successfully!'
        })
        
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global multi_emotion_model
    
    return jsonify({
        'status': 'healthy',
        'multi_emotion_model_loaded': multi_emotion_model is not None,
        'method': 'multi_emotion_classification',
        'adjustable_weights': True,
        'emotion_count': 7
    })

if __name__ == '__main__':
    print("🚀 Starting Multi-Emotion Happiness Detection Server...")
    print("="*70)
    print("🎭 Method: Multi-emotion classification → Happiness calculation")
    print("⚖️ Adjustable weights: Change happiness calculation without retraining!")
    print("="*70)
    
    # Load multi-emotion model
    model_loaded = load_multi_emotion_model()
    
    if model_loaded:
        print("✅ Multi-emotion model loaded successfully!")
        print("🎯 Ready for emotion classification and happiness calculation")
    else:
        print("❌ Multi-emotion model not found")
        print("💡 Train model first: python train_multi_emotion.py")
    
    print("🌐 Server starting on http://localhost:5000")
    print("📡 Endpoints: /predict, /adjust_weights, /health")
    print("="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)