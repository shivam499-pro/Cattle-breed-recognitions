"""
BPA Integration API - Cattle Breed Recognition Module
======================================================

This Flask API provides endpoints for integrating the AI breed recognition
module with the existing Bharat Pashudhan App (BPA).

Two-Stage Pipeline:
  Stage 1: YOLOv8-Nano - Detect cattle in image
  Stage 2: EfficientNet-B0 - Classify breed (41 breeds)

Author: SIH 2025 Team
Problem Statement: SIH25004
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import json
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for BPA integration

# Configuration
CONFIG = {
    'MODELS_DIR': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
    'CONFIDENCE_THRESHOLD': 0.85,  # Auto-confirm threshold
    'ESCALATION_THRESHOLD': 0.60,  # FLW select threshold
    'MAX_IMAGE_SIZE': 10 * 1024 * 1024,  # 10MB max
    'SUPPORTED_FORMATS': ['jpg', 'jpeg', 'png', 'webp']
}


class BreedRecognitionEngine:
    """
    Main AI engine for breed recognition.
    Uses YOLOv8-Nano for detection and EfficientNet-B0 for classification.
    """
    
    def __init__(self, models_dir):
        """Initialize the AI engine with trained models."""
        self.models_dir = models_dir
        self.yolo_model = None
        self.classifier = None
        self.breed_mapping = {}
        self.idx_to_breed = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models for detection and classification."""
        # Suppress TF warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load YOLOv8-Nano detector
        yolo_path = os.path.join(self.models_dir, 'cattle_detector.pt')
        if os.path.exists(yolo_path):
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_path)
            logger.info(f"✅ Loaded YOLOv8-Nano detector: {yolo_path}")
        else:
            logger.warning(f"❌ YOLO model not found: {yolo_path}")
        
        # Load EfficientNet-B0 classifier
        classifier_path = os.path.join(self.models_dir, 'breed_classifier_v2.keras')
        if os.path.exists(classifier_path):
            from tensorflow.keras.models import load_model
            self.classifier = load_model(classifier_path)
            logger.info(f"✅ Loaded EfficientNet-B0 classifier: {classifier_path}")
        else:
            logger.warning(f"❌ Classifier not found: {classifier_path}")
        
        # Load breed mapping
        mapping_path = os.path.join(self.models_dir, 'breed_mapping_v2.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.breed_mapping = json.load(f)
            self.idx_to_breed = {int(k): v for k, v in self.breed_mapping.items()}
            logger.info(f"✅ Loaded breed mapping: {len(self.idx_to_breed)} breeds")
        else:
            logger.warning(f"❌ Breed mapping not found: {mapping_path}")
    
    def detect_cattle(self, image_array, conf_threshold=0.5):
        """
        Detect cattle in image using YOLOv8-Nano.
        
        Args:
            image_array: numpy array (BGR format from cv2)
            conf_threshold: Minimum confidence for detection
            
        Returns:
            List of detections with bbox and crop
        """
        if self.yolo_model is None:
            # Return full image as fallback
            h, w = image_array.shape[:2]
            return [{
                'detected': True,
                'bbox': [0, 0, w, h],
                'confidence': 1.0,
                'crop': image_array
            }]
        
        # Convert BGR to RGB for YOLO
        image_rgb = image_array[:, :, ::-1]
        
        # Run YOLO detection
        results = self.yolo_model.predict(image_rgb, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Crop detected region
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = image_array[y1:y2, x1:x2]
                
                detections.append({
                    'detected': True,
                    'bbox': [x1, y1, x2-x1, y2-y1],  # x, y, width, height
                    'confidence': float(conf),
                    'crop': crop
                })
        
        return detections if detections else [{
            'detected': False,
            'bbox': None,
            'confidence': 0.0,
            'crop': None
        }]
    
    def classify_breed(self, image_crop):
        """
        Classify breed using EfficientNet-B0.
        
        Args:
            image_crop: numpy array (BGR format)
            
        Returns:
            dict: Classification results with breed, confidence, and top predictions
        """
        if self.classifier is None or image_crop is None:
            return {
                'breed': 'Unknown',
                'confidence': 0.0,
                'top_predictions': []
            }
        
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            
            # Convert BGR to RGB
            image_rgb = image_crop[:, :, ::-1]
            
            # Resize to 224x224
            pil_image = Image.fromarray(image_rgb)
            pil_image = pil_image.resize((224, 224))
            
            # Convert to array and preprocess
            img_array = np.array(pil_image)
            if img_array.shape[-1] == 4:  # RGBA to RGB
                img_array = img_array[:, :, :3]
            img_array = preprocess_input(img_array)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.classifier.predict(img_batch, verbose=0)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = []
            for idx in top_3_indices:
                top_predictions.append({
                    'breed': self.idx_to_breed.get(idx, f'Unknown_{idx}'),
                    'confidence': float(predictions[idx])
                })
            
            return {
                'breed': self.idx_to_breed.get(top_3_indices[0], 'Unknown'),
                'confidence': float(predictions[top_3_indices[0]]),
                'top_predictions': top_predictions
            }
        
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                'breed': 'Error',
                'confidence': 0.0,
                'top_predictions': [],
                'error': str(e)
            }
    
    def predict(self, image_array):
        """
        Full prediction pipeline: Detection -> Classification.
        
        Args:
            image_array: numpy array of image (BGR format)
        
        Returns:
            dict: Complete prediction results
        """
        import time
        start_time = time.time()
        
        # Stage 1: Detection
        detections = self.detect_cattle(image_array)
        
        results = []
        for detection in detections:
            if detection['detected'] and detection['crop'] is not None:
                # Stage 2: Classification
                classification = self.classify_breed(detection['crop'])
                
                # Determine action based on confidence
                confidence = classification['confidence']
                if confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
                    action = 'auto_confirm'
                elif confidence >= CONFIG['ESCALATION_THRESHOLD']:
                    action = 'flw_select'
                else:
                    action = 'expert_review'
                
                results.append({
                    'detection': {
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    },
                    'classification': classification,
                    'action': action
                })
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'success': len(results) > 0,
            'num_cattle': len(results),
            'results': results,
            'inference_time_ms': round(inference_time, 2),
            'timestamp': datetime.now().isoformat()
        }


# Initialize AI engine
ai_engine = BreedRecognitionEngine(CONFIG['MODELS_DIR'])


# ==================== API Routes ====================

@app.route('/', methods=['GET'])
def index():
    """API documentation page."""
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_info():
    """API information."""
    return jsonify({
        'name': 'Cattle Breed Recognition API',
        'version': '2.0.0',
        'description': 'AI-powered breed recognition for Bharat Pashudhan App',
        'models': {
            'detection': 'YOLOv8-Nano (99.5% mAP)',
            'classification': 'EfficientNet-B0 (72.61% accuracy, 41 breeds)'
        },
        'endpoints': {
            'POST /predict': 'Predict breed from uploaded image',
            'POST /predict/base64': 'Predict breed from base64 image',
            'GET /breeds': 'List all supported breeds',
            'GET /health': 'Health check',
            'POST /feedback': 'Submit feedback for model improvement',
            'POST /escalate': 'Escalate to expert review'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'detector': ai_engine.yolo_model is not None,
            'classifier': ai_engine.classifier is not None
        },
        'breeds_count': len(ai_engine.idx_to_breed),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/breeds', methods=['GET'])
def list_breeds():
    """List all supported breeds."""
    breeds = sorted(ai_engine.idx_to_breed.values())
    
    # Categorize breeds
    cattle_breeds = ['Alambadi', 'Amritmahal', 'Bargur', 'Dangi', 'Deoni', 'Gir', 
                     'Hallikar', 'Hariana', 'Kangayam', 'Kankrej', 'Kasargod',
                     'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley',
                     'Malnad_gidda', 'Nagori', 'Nagpuri', 'Nimari', 'Ongole',
                     'Pulikulam', 'Rathi', 'Red_Sindhi', 'Sahiwal', 'Tharparkar',
                     'Toda', 'Umblachery', 'Vechur']
    buffalo_breeds = ['Banni', 'Bhadawari', 'Jaffrabadi', 'Mehsana', 'Murrah',
                      'Nili_Ravi', 'Surti']
    foreign_breeds = ['Ayrshire', 'Brown_Swiss', 'Guernsey', 'Holstein_Friesian',
                      'Jersey', 'Red_Dane']
    
    return jsonify({
        'total': len(breeds),
        'cattle_indian': [b for b in breeds if b in cattle_breeds],
        'buffalo_indian': [b for b in breeds if b in buffalo_breeds],
        'foreign': [b for b in breeds if b in foreign_breeds],
        'all': breeds
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict breed from uploaded image.
    
    Expects multipart/form-data with 'image' file.
    
    Returns:
        JSON with breed prediction, confidence, and action
    """
    # Check for image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Use multipart/form-data with "image" field.'}), 400
    
    file = request.files['image']
    
    # Validate file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in CONFIG['SUPPORTED_FORMATS']:
        return jsonify({'error': f'Unsupported format. Use: {CONFIG["SUPPORTED_FORMATS"]}'}), 400
    
    try:
        # Read image
        import cv2
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run prediction
        result = ai_engine.predict(image)
        
        # Add request metadata
        result['request_id'] = request.headers.get('X-Request-ID', 'unknown')
        result['flw_id'] = request.headers.get('X-FLW-ID', 'unknown')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """
    Predict breed from base64 encoded image.
    
    Expects JSON body:
    {
        "image": "base64_encoded_image",
        "animal_id": "optional_animal_id",
        "flw_id": "optional_flw_id"
    }
    
    Returns:
        JSON with breed prediction, confidence, and action
    """
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided. Send JSON with "image" field.'}), 400
    
    try:
        # Decode base64
        import cv2
        image_data = base64.b64decode(data['image'])
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Run prediction
        result = ai_engine.predict(image)
        
        # Add metadata
        result['animal_id'] = data.get('animal_id', 'unknown')
        result['flw_id'] = data.get('flw_id', 'unknown')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback for model improvement.
    
    This endpoint collects FLW/expert corrections to improve the model
    through the self-learning feedback loop.
    
    Expects JSON body:
    {
        "prediction_id": "unique_id",
        "predicted_breed": "Gir",
        "actual_breed": "Sahiwal",
        "confidence": 0.75,
        "flw_id": "FLW001",
        "expert_verified": false
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No feedback data provided'}), 400
    
    required_fields = ['predicted_breed', 'actual_breed']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Store feedback for model improvement
    feedback_record = {
        'prediction_id': data.get('prediction_id', 'unknown'),
        'predicted_breed': data['predicted_breed'],
        'actual_breed': data['actual_breed'],
        'confidence': data.get('confidence', 0.0),
        'flw_id': data.get('flw_id', 'unknown'),
        'expert_verified': data.get('expert_verified', False),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to feedback file
    feedback_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'feedback'))
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback_record) + '\n')
    
    # Check if prediction was correct
    is_correct = data['predicted_breed'] == data['actual_breed']
    
    return jsonify({
        'status': 'success',
        'message': 'Feedback recorded for model improvement',
        'is_correct': is_correct,
        'timestamp': feedback_record['timestamp']
    })


@app.route('/escalate', methods=['POST'])
def escalate_to_expert():
    """
    Escalate uncertain case to expert review.
    
    Expects JSON body:
    {
        "prediction_id": "unique_id",
        "image_base64": "base64_image",
        "top_predictions": [...],
        "flw_notes": "Optional notes from FLW",
        "animal_id": "Pashu Aadhaar ID"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No escalation data provided'}), 400
    
    # Create escalation record
    escalation = {
        'escalation_id': f"ESC_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'prediction_id': data.get('prediction_id', 'unknown'),
        'top_predictions': data.get('top_predictions', []),
        'flw_notes': data.get('flw_notes', ''),
        'animal_id': data.get('animal_id', 'unknown'),
        'status': 'pending',
        'assigned_to': None,
        'created_at': datetime.now().isoformat()
    }
    
    # Save escalation
    escalation_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'escalations'))
    escalation_dir.mkdir(parents=True, exist_ok=True)
    
    escalation_file = escalation_dir / f"{escalation['escalation_id']}.json"
    with open(escalation_file, 'w') as f:
        json.dump(escalation, f, indent=2)
    
    # Save image separately
    if 'image_base64' in data:
        image_dir = escalation_dir / 'images'
        image_dir.mkdir(exist_ok=True)
        image_file = image_dir / f"{escalation['escalation_id']}.jpg"
        with open(image_file, 'wb') as f:
            f.write(base64.b64decode(data['image_base64']))
    
    return jsonify({
        'status': 'success',
        'message': 'Case escalated to expert review',
        'escalation_id': escalation['escalation_id'],
        'estimated_response_time': '24 hours'
    })


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Expert dashboard for reviewing escalated cases."""
    escalation_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'escalations'))
    
    pending_cases = []
    if escalation_dir.exists():
        for file in escalation_dir.glob('ESC_*.json'):
            with open(file, 'r') as f:
                case = json.load(f)
                if case.get('status') == 'pending':
                    pending_cases.append(case)
    
    return render_template('dashboard.html', cases=pending_cases)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🐄 Cattle Breed Recognition API")
    print("="*60)
    print(f"\nModels Directory: {CONFIG['MODELS_DIR']}")
    print(f"Breeds Supported: {len(ai_engine.idx_to_breed)}")
    print(f"\nEndpoints:")
    print(f"  - http://localhost:5000/")
    print(f"  - http://localhost:5000/predict (POST)")
    print(f"  - http://localhost:5000/predict/base64 (POST)")
    print(f"  - http://localhost:5000/breeds (GET)")
    print(f"  - http://localhost:5000/health (GET)")
    print(f"  - http://localhost:5000/dashboard (GET)")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
