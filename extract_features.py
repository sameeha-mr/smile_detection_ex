import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def extract_features(image_path, cnn_smile_score):
    """
    Extract facial features from image using MediaPipe
    Returns a dictionary of features for fuzzy logic processing
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                logger.warning("No face detected in image")
                return None
            
            # Get the first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key facial points
            h, w = image.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            
            # Extract specific features
            features = extract_facial_features(landmarks, cnn_smile_score)
            
            return features
            
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def extract_facial_features(landmarks, cnn_smile_score):
    """
    Extract specific facial features from MediaPipe landmarks
    """
    try:
        # Key landmark indices for facial features
        # Mouth corners
        left_mouth_corner = landmarks[61]   # Left corner
        right_mouth_corner = landmarks[291] # Right corner
        
        # Mouth top and bottom
        upper_lip_top = landmarks[13]       # Top of upper lip
        lower_lip_bottom = landmarks[14]    # Bottom of lower lip
        
        # Eyes
        left_eye_outer = landmarks[33]      # Left eye outer corner
        left_eye_inner = landmarks[133]     # Left eye inner corner
        right_eye_outer = landmarks[362]    # Right eye outer corner
        right_eye_inner = landmarks[463]    # Right eye inner corner
        
        # Eye centers (approximate)
        left_eye_center = (landmarks[159] + landmarks[145]) // 2
        right_eye_center = (landmarks[386] + landmarks[374]) // 2
        
        # Eyebrows
        left_eyebrow_inner = landmarks[70]
        left_eyebrow_outer = landmarks[46]
        right_eyebrow_inner = landmarks[300]
        right_eyebrow_outer = landmarks[276]
        
        # Calculate distances and ratios
        
        # 1. Mouth width
        mouth_width = np.linalg.norm(right_mouth_corner - left_mouth_corner)
        
        # 2. Mouth height (lip separation)
        mouth_height = np.linalg.norm(upper_lip_top - lower_lip_bottom)
        
        # 3. Mouth aspect ratio
        # mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # 4. Eye distances
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # 5. Eye-mouth ratio
        eye_mouth_ratio = avg_eye_width / mouth_width if mouth_width > 0 else 0
        
        # 6. Smile curvature (approximate)
        # Calculate the angle of mouth corners relative to mouth center
        mouth_center = (left_mouth_corner + right_mouth_corner) // 2
        left_angle = np.arctan2(left_mouth_corner[1] - mouth_center[1], 
                               left_mouth_corner[0] - mouth_center[0])
        right_angle = np.arctan2(right_mouth_corner[1] - mouth_center[1], 
                                right_mouth_corner[0] - mouth_center[0])
        
        smile_curvature = abs(left_angle - right_angle) * 180 / np.pi
        
        # 7. Eyebrow position (happiness indicator)
        left_brow_height = left_eyebrow_inner[1] - left_eye_center[1]
        right_brow_height = right_eyebrow_inner[1] - right_eye_center[1]
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        # 8. Face width (for normalization)
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])  # Face outline points
        
        # Normalize features by face width
        normalized_mouth_width = mouth_width / face_width if face_width > 0 else 0
        normalized_mouth_height = mouth_height / face_width if face_width > 0 else 0
        normalized_eye_width = avg_eye_width / face_width if face_width > 0 else 0
        
        # Create feature dictionary
        features = {
            'cnn_smile_score': float(cnn_smile_score),
            'mouth_width': float(normalized_mouth_width),
            'mouth_height': float(normalized_mouth_height),
            #'mouth_aspect_ratio': float(mouth_aspect_ratio),
            'eye_mouth_ratio': float(eye_mouth_ratio),
            'smile_curvature': float(smile_curvature),
            'eyebrow_height': float(avg_brow_height / face_width if face_width > 0 else 0),
            'eye_width': float(normalized_eye_width)
        }
        
        logger.info(f"Extracted features: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error calculating facial features: {e}")
        return None