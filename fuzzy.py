import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

logger = logging.getLogger(__name__)

def get_happiness_level_and_emoji(score):
    """Determine happiness level and emoji based on score"""
    if score >= 80:
        return "Very Happy", "😄", "Extremely positive emotion detected!"
    elif score >= 65:
        return "Happy", "😊", "Clear signs of happiness"
    elif score >= 50:
        return "Somewhat Happy", "🙂", "Mild positive emotion"
    elif score >= 35:
        return "Neutral", "😐", "Balanced emotional state"
    elif score >= 20:
        return "Somewhat Sad", "😕", "Mild negative emotion"
    elif score >= 10:
        return "Sad", "😔", "Some signs of sadness detected"
    else:
        return "Very Sad", "😢", "Strong negative emotion detected"

def create_fuzzy_system():
    """Create WORKING fuzzy logic system - NO undefined variables!"""
    try:
        # ONLY 6 INPUT VARIABLES (removed problematic mouth_aspect_ratio)
        cnn_score = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cnn_score')
        mouth_width = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'mouth_width')
        mouth_height = ctrl.Antecedent(np.arange(0, 2.0, 0.01), 'mouth_height')
        eye_mouth_ratio = ctrl.Antecedent(np.arange(0, 3.0, 0.01), 'eye_mouth_ratio')
        smile_curvature = ctrl.Antecedent(np.arange(0, 180, 1), 'smile_curvature')
        eyebrow_height = ctrl.Antecedent(np.arange(-0.5, 0.5, 0.001), 'eyebrow_height')
        
        # Output variable
        happiness = ctrl.Consequent(np.arange(0, 101, 1), 'happiness')
        
        # CNN Score membership functions
        cnn_score['very_low'] = fuzz.trapmf(cnn_score.universe, [0, 0, 0.15, 0.3])
        cnn_score['low'] = fuzz.trapmf(cnn_score.universe, [0.2, 0.3, 0.4, 0.5])
        cnn_score['medium'] = fuzz.trapmf(cnn_score.universe, [0.4, 0.5, 0.6, 0.7])
        cnn_score['high'] = fuzz.trapmf(cnn_score.universe, [0.6, 0.7, 0.8, 0.9])
        cnn_score['very_high'] = fuzz.trapmf(cnn_score.universe, [0.8, 0.9, 1, 1])
        
        # Mouth width membership functions
        mouth_width['narrow'] = fuzz.trapmf(mouth_width.universe, [0, 0, 0.4, 0.8])
        mouth_width['medium'] = fuzz.trapmf(mouth_width.universe, [0.6, 1.0, 1.4, 1.8])
        mouth_width['wide'] = fuzz.trapmf(mouth_width.universe, [1.6, 2.0, 2.0, 2.0])
        
        # Mouth height membership functions
        mouth_height['closed'] = fuzz.trapmf(mouth_height.universe, [0, 0, 0.3, 0.6])
        mouth_height['slightly_open'] = fuzz.trapmf(mouth_height.universe, [0.4, 0.7, 1.0, 1.3])
        mouth_height['open'] = fuzz.trapmf(mouth_height.universe, [1.1, 1.4, 2.0, 2.0])
        
        # Eye-mouth ratio membership functions
        eye_mouth_ratio['small'] = fuzz.trapmf(eye_mouth_ratio.universe, [0, 0, 0.8, 1.2])
        eye_mouth_ratio['normal'] = fuzz.trapmf(eye_mouth_ratio.universe, [1.0, 1.4, 1.8, 2.2])
        eye_mouth_ratio['large'] = fuzz.trapmf(eye_mouth_ratio.universe, [2.0, 2.4, 3.0, 3.0])
        
        # Smile curvature membership functions
        smile_curvature['low'] = fuzz.trapmf(smile_curvature.universe, [0, 20, 60, 100])
        smile_curvature['medium'] = fuzz.trapmf(smile_curvature.universe, [80, 100, 120, 140])
        smile_curvature['high'] = fuzz.trapmf(smile_curvature.universe, [130, 150, 180, 180])
        
        # Eyebrow height membership functions
        eyebrow_height['lowered'] = fuzz.trapmf(eyebrow_height.universe, [-0.5, -0.3, -0.1, -0.05])
        eyebrow_height['neutral'] = fuzz.trapmf(eyebrow_height.universe, [-0.08, -0.02, 0.02, 0.08])
        eyebrow_height['raised'] = fuzz.trapmf(eyebrow_height.universe, [0.05, 0.1, 0.3, 0.5])
        
        # Output happiness membership functions
        happiness['very_sad'] = fuzz.trapmf(happiness.universe, [0, 0, 8, 18])
        happiness['sad'] = fuzz.trapmf(happiness.universe, [12, 20, 28, 36])
        happiness['somewhat_sad'] = fuzz.trapmf(happiness.universe, [30, 35, 40, 45])
        happiness['neutral'] = fuzz.trapmf(happiness.universe, [40, 45, 55, 60])
        happiness['somewhat_happy'] = fuzz.trapmf(happiness.universe, [55, 60, 65, 70])
        happiness['happy'] = fuzz.trapmf(happiness.universe, [65, 72, 78, 85])
        happiness['very_happy'] = fuzz.trapmf(happiness.universe, [80, 88, 100, 100])
        
        # SIMPLIFIED RULES (using only the 6 defined inputs)
        rules = [
            # Very Happy Rules
            ctrl.Rule(cnn_score['very_high'] & mouth_width['wide'] & smile_curvature['high'], happiness['very_happy']),
            ctrl.Rule(cnn_score['high'] & mouth_width['wide'] & smile_curvature['high'], happiness['very_happy']),
            
            # Happy Rules
            ctrl.Rule(cnn_score['high'] & mouth_width['medium'], happiness['happy']),
            ctrl.Rule(cnn_score['medium'] & mouth_width['wide'], happiness['happy']),
            ctrl.Rule(cnn_score['high'] & smile_curvature['medium'], happiness['happy']),
            
            # Somewhat Happy Rules
            ctrl.Rule(cnn_score['medium'] & mouth_width['medium'], happiness['somewhat_happy']),
            ctrl.Rule(cnn_score['high'] & mouth_height['open'], happiness['somewhat_happy']),
            
            # Neutral Rules
            ctrl.Rule(cnn_score['medium'] & eyebrow_height['neutral'], happiness['neutral']),
            ctrl.Rule(cnn_score['low'] & mouth_width['medium'], happiness['neutral']),
            
            # Somewhat Sad Rules
            ctrl.Rule(cnn_score['low'] & mouth_width['narrow'], happiness['somewhat_sad']),
            ctrl.Rule(cnn_score['medium'] & eyebrow_height['lowered'], happiness['somewhat_sad']),
            
            # Sad Rules
            ctrl.Rule(cnn_score['low'] & eyebrow_height['lowered'], happiness['sad']),
            ctrl.Rule(cnn_score['very_low'] & smile_curvature['low'], happiness['sad']),
            
            # Very Sad Rules
            ctrl.Rule(cnn_score['very_low'] & mouth_width['narrow'], happiness['very_sad']),
            ctrl.Rule(cnn_score['very_low'] & eyebrow_height['lowered'], happiness['very_sad']),
            
            # Single feature dominance
            ctrl.Rule(cnn_score['very_high'], happiness['happy']),
            ctrl.Rule(cnn_score['very_low'], happiness['sad']),
        ]
        
        # Create control system
        happiness_ctrl = ctrl.ControlSystem(rules)
        happiness_sim = ctrl.ControlSystemSimulation(happiness_ctrl)
        
        logger.info(f"Created FIXED fuzzy system with {len(rules)} rules")
        return happiness_sim
        
    except Exception as e:
        logger.error(f"Error creating fuzzy system: {e}")
        return None

def compute_happiness(features):
    """COMPLETELY FIXED fuzzy logic computation"""
    try:
        fuzzy_system = create_fuzzy_system()
        if fuzzy_system is None:
            return enhanced_fallback_calculation(features)
        
        # FIXED: Better handling of extreme values
        cnn_score = features.get('cnn_smile_score', 0.5)
        
        # Handle scientific notation and extreme values
        if cnn_score < 1e-10:  # If essentially zero (like 2.19e-15)
            cnn_score = 0.0
            logger.warning(f"CNN score too small ({features.get('cnn_smile_score')}), set to 0.0")
        
        cnn_score = max(0, min(1, cnn_score))
        
        # FIXED: More robust scaling with bounds checking
        mouth_width = features.get('mouth_width', 0.1)
        mouth_height = features.get('mouth_height', 0.05)
        eye_mouth_ratio = features.get('eye_mouth_ratio', 0.5)
        smile_curvature = features.get('smile_curvature', 90)
        eyebrow_height = features.get('eyebrow_height', 0)
        
        # Handle zero mouth height
        if mouth_height == 0.0:
            mouth_height = 0.01  # Small default
            logger.warning("Mouth height was 0.0, set to 0.01")
        
        # Scale features
        scaled_mouth_width = max(0, min(2, mouth_width * 3))
        scaled_mouth_height = max(0, min(2, mouth_height * 15))
        scaled_eye_mouth_ratio = max(0, min(3, eye_mouth_ratio * 2))
        scaled_smile_curvature = max(0, min(180, smile_curvature))
        scaled_eyebrow_height = max(-0.5, min(0.5, eyebrow_height * 5))
        
        # Debug logging
        logger.info(f"FIXED INPUTS:")
        logger.info(f"  CNN: {cnn_score:.3f} (orig: {features.get('cnn_smile_score')})")
        logger.info(f"  MouthW: {scaled_mouth_width:.3f} (orig: {mouth_width:.3f})")
        logger.info(f"  MouthH: {scaled_mouth_height:.3f} (orig: {mouth_height:.3f})")
        logger.info(f"  EyeMouthR: {scaled_eye_mouth_ratio:.3f} (orig: {eye_mouth_ratio:.3f})")
        logger.info(f"  Curvature: {scaled_smile_curvature:.1f}")
        logger.info(f"  EyebrowH: {scaled_eyebrow_height:.3f} (orig: {eyebrow_height:.3f})")
        
        # Set inputs (ONLY the 6 that are actually defined!)
        fuzzy_system.input['cnn_score'] = cnn_score
        fuzzy_system.input['mouth_width'] = scaled_mouth_width
        fuzzy_system.input['mouth_height'] = scaled_mouth_height
        fuzzy_system.input['eye_mouth_ratio'] = scaled_eye_mouth_ratio
        fuzzy_system.input['smile_curvature'] = scaled_smile_curvature
        fuzzy_system.input['eyebrow_height'] = scaled_eyebrow_height
        
        # Compute
        fuzzy_system.compute()
        happiness_score = fuzzy_system.output['happiness']
        
        # Get classification
        happiness_level, emoji, description = get_happiness_level_and_emoji(happiness_score)
        
        logger.info(f"FUZZY SUCCESS: {happiness_score:.1f}% -> {happiness_level} {emoji}")
        
        return {
            'happiness_score': float(happiness_score),
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description
        }
        
    except Exception as e:
        logger.error(f"Fuzzy computation failed: {e}")
        logger.error(f"Available features: {list(features.keys())}")
        return enhanced_fallback_calculation(features)

def enhanced_fallback_calculation(features):
    """Enhanced fallback calculation with better extreme value handling"""
    try:
        cnn_score = features.get('cnn_smile_score', 0.5)
        mouth_width = features.get('mouth_width', 0.1)
        mouth_height = features.get('mouth_height', 0.05)
        curvature = features.get('smile_curvature', 90)
        eyebrow_height = features.get('eyebrow_height', 0)
        
        # Handle extreme CNN score
        if cnn_score < 1e-10:  # Scientific notation handling
            cnn_score = 0.0
        
        # Handle zero mouth height
        if mouth_height == 0.0:
            mouth_height = 0.01
        
        # Enhanced calculation with better scaling
        base_score = cnn_score * 50  # Reduced impact when CNN is 0
        width_bonus = min(25, mouth_width * 80)  # Width contribution
        height_bonus = min(15, mouth_height * 150)  # Height contribution
        curvature_bonus = max(-5, min(20, (curvature - 120) * 0.2))  # Curvature
        eyebrow_bonus = max(-20, min(10, eyebrow_height * 100))  # Eyebrow
        
        # Special handling for all-zero case
        if cnn_score == 0 and mouth_height <= 0.01 and mouth_width < 0.2:
            base_score = 20  # Give minimal baseline for detection
        
        final_score = base_score + width_bonus + height_bonus + curvature_bonus + eyebrow_bonus
        final_score = max(5, min(95, final_score))  # Keep in reasonable range
        
        happiness_level, emoji, description = get_happiness_level_and_emoji(final_score)
        
        logger.info(f"ENHANCED FALLBACK: {final_score:.1f}% "
                   f"(Base:{base_score:.1f} + Width:{width_bonus:.1f} + "
                   f"Height:{height_bonus:.1f} + Curve:{curvature_bonus:.1f} + Brow:{eyebrow_bonus:.1f})")
        
        return {
            'happiness_score': float(final_score),
            'happiness_level': happiness_level,
            'emoji': emoji,
            'description': description
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced fallback: {e}")
        return {
            'happiness_score': 30.0,
            'happiness_level': 'Neutral',
            'emoji': '😐',
            'description': 'Unable to determine emotion accurately'
        }

# Test function
def test_extreme_cases():
    """Test the fuzzy system with extreme cases"""
    print("🧪 Testing Extreme Cases...")
    
    test_cases = [
        {
            'name': 'Zero CNN Score (like your log)',
            'features': {
                'cnn_smile_score': 2.199040467882013e-15,  # Your actual value
                'mouth_width': 0.32786885245901637,
                'mouth_height': 0.0,
                'eye_mouth_ratio': 0.35096012026013246,
                'smile_curvature': 180.0,
                'eyebrow_height': -0.13114754098360656,
                'eye_width': 0.11506889188856802
            }
        },
        {
            'name': 'Normal Happy Case',
            'features': {
                'cnn_smile_score': 0.8,
                'mouth_width': 0.5,
                'mouth_height': 0.1,
                'eye_mouth_ratio': 0.3,
                'smile_curvature': 175,
                'eyebrow_height': 0.05,
                'eye_width': 0.12
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Testing {test_case['name']}:")
        result = compute_happiness(test_case['features'])
        print(f"   Result: {result['happiness_score']:.1f}% - {result['happiness_level']} {result['emoji']}")

if __name__ == "__main__":
    test_extreme_cases()