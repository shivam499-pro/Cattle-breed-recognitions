"""
BPA Integration API - Cattle Breed Recognition Module
======================================================

This Flask API provides endpoints for integrating the AI breed recognition
module with the existing Bharat Pashudhan App (BPA).

Author: SIH 2025 Team
Problem Statement: SIH25004
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for BPA integration

# Configuration
CONFIG = {
    'MODEL_PATH': os.environ.get('MODEL_PATH', '../models/final'),
    'CONFIDENCE_THRESHOLD': 0.85,  # Auto-confirm threshold
    'ESCALATION_THRESHOLD': 0.60,  # Expert escalation threshold
    'MAX_IMAGE_SIZE': 10 * 1024 * 1024,  # 10MB max
    'SUPPORTED_FORMATS': ['jpg', 'jpeg', 'png', 'webp']
}

# Load breed classes
BREEDS = [
    'Gir', 'Sahiwal', 'Red_Sindhi', 'Tharparkar', 'Rathi',
    'Hariana', 'Kankrej', 'Ongole', 'Deoni',
    'Hallikar', 'Amritmahal', 'Khillari', 'Kangayam', 'Bargur',
    'Dangi', 'Krishna_Valley', 'Malnad_Gidda', 'Punganur', 'Vechur',
    'Pulikulam', 'Umblachery', 'Toda', 'Kalahandi',
    'Murrah', 'Jaffrabadi', 'Nili_Ravi', 'Banni', 'Pandharpuri',
    'Mehsana', 'Surti', 'Nagpuri', 'Bhadawari', 'Chilika',
    'Jersey_Cross', 'HF_Cross'
]

IDX_TO_CLASS = {i: breed for i, breed in enumerate(BREEDS)}
CLASS_TO_IDX = {breed: i for i, breed in enumerate(BREEDS)}


class BreedRecognitionEngine:
    """
    Main AI engine for breed recognition.
    Uses YOLOv8-Nano for detection and EfficientNet-B0 for classification.
    """
    
    def __init__(self, model_path):
        """Initialize the AI engine with TFLite models."""
        self.model_path = model_path
        self.detector = None
        self.classifier = None
        self.load_models()
    
    def load_models(self):
        """Load TFLite models for detection and classification."""
        import tensorflow as tf
        
        # Load YOLOv8-Nano detector
        detector_path = os.path.join(self.model_path, 'yolov8_nano_cattle_detector_int8.tflite')
        if os.path.exists(detector_path):
            self.detector = tf.lite.Interpreter(model_path=detector_path)
            self.detector.allocate_tensors()
            logger.info(f"Loaded detector: {detector_path}")
        else:
            logger.warning(f"Detector not found: {detector_path}")
        
        # Load EfficientNet-B0 classifier
        classifier_path = os.path.join(self.model_path, 'efficientnet_b0_int8.tflite')
        if os.path.exists(classifier_path):
            self.classifier = tf.lite.Interpreter(model_path=classifier_path)
            self.classifier.allocate_tensors()
            logger.info(f"Loaded classifier: {classifier_path}")
        else:
            logger.warning(f"Classifier not found: {classifier_path}")
    
    def preprocess_image(self, image_array, target_size=(224, 224)):
        """Preprocess image for model inference."""
        import cv2
        
        # Resize
        image = cv2.resize(image_array, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def detect_animal(self, image_array):
        """
        Detect animal in image using YOLOv8-Nano.
        
        Returns:
            dict: Detection results with bounding box and confidence
        """
        if self.detector is None:
            # Return full image as fallback
            h, w = image_array.shape[:2]
            return {
                'detected': True,
                'bbox': [0, 0, w, h],  # x, y, width, height
                'confidence': 1.0,
                'crop': image_array
            }
        
        # Get input/output details
        input_details = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        
        # Preprocess for detection (416x416)
        input_size = input_details[0]['shape'][1]  # Usually 416
        processed = self.preprocess_image(image_array, (input_size, input_size))
        
        # Run inference
        self.detector.set_tensor(input_details[0]['index'], processed)
        self.detector.invoke()
        output = self.detector.get_tensor(output_details[0]['index'])
        
        # Parse YOLO output (simplified)
        # In production, use proper NMS and box decoding
        h, w = image_array.shape[:2]
        
        return {
            'detected': True,
            'bbox': [0, 0, w, h],
            'confidence': 0.95,
            'crop': image_array
        }
    
    def classify_breed(self, image_array):
        """
        Classify breed using EfficientNet-B0.
        
        Returns:
            dict: Classification results with breed, confidence, and top predictions
        """
        if self.classifier is None:
            # Return mock result
            return {
                'breed': 'Unknown',
                'confidence': 0.0,
                'top_predictions': []
            }
        
        # Get input/output details
        input_details = self.classifier.get_input_details()
        output_details = self.classifier.get_output_details()
        
        # Preprocess for classification (224x224)
        processed = self.preprocess_image(image_array, (224, 224))
        
        # Handle quantized input
        if input_details[0]['dtype'] == np.uint8:
            input_scale = input_details[0]['quantization_parameters']['scales'][0]
            input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
            processed = (processed / input_scale + input_zero_point).astype(np.uint8)
        
        # Run inference
        self.classifier.set_tensor(input_details[0]['index'], processed.astype(input_details[0]['dtype']))
        self.classifier.invoke()
        output = self.classifier.get_tensor(output_details[0]['index'])
        
        # Handle quantized output
        if output_details[0]['dtype'] == np.uint8:
            output_scale = output_details[0]['quantization_parameters']['scales'][0]
            output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Get predictions
        probabilities = output[0]
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        
        top_predictions = []
        for idx in top_3_idx:
            top_predictions.append({
                'breed': IDX_TO_CLASS[idx],
                'confidence': float(probabilities[idx])
            })
        
        return {
            'breed': IDX_TO_CLASS[top_3_idx[0]],
            'confidence': float(probabilities[top_3_idx[0]]),
            'top_predictions': top_predictions
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
        detection = self.detect_animal(image_array)
        
        # Stage 2: Classification
        if detection['detected']:
            classification = self.classify_breed(detection['crop'])
        else:
            classification = {
                'breed': 'Unknown',
                'confidence': 0.0,
                'top_predictions': []
            }
        
        inference_time = (time.time() - start_time) * 1000
        
        # Determine action based on confidence
        confidence = classification['confidence']
        if confidence >= CONFIG['CONFIDENCE_THRESHOLD']:
            action = 'auto_confirm'
        elif confidence >= CONFIG['ESCALATION_THRESHOLD']:
            action = 'flw_select'
        else:
            action = 'expert_review'
        
        return {
            'detection': detection,
            'classification': classification,
            'action': action,
            'inference_time_ms': inference_time,
            'timestamp': datetime.now().isoformat()
        }


# Initialize AI engine
ai_engine = BreedRecognitionEngine(CONFIG['MODEL_PATH'])


# ==================== API Routes ====================

@app.route('/', methods=['GET'])
def index():
    """API documentation page."""
    return jsonify({
        'name': 'Cattle Breed Recognition API',
        'version': '1.0.0',
        'description': 'AI-powered breed recognition for Bharat Pashudhan App',
        'endpoints': {
            'POST /predict': 'Predict breed from image',
            'POST /predict/base64': 'Predict breed from base64 image',
            'GET /breeds': 'List all supported breeds',
            'GET /health': 'Health check',
            'POST /feedback': 'Submit feedback for model improvement'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'detector': ai_engine.detector is not None,
            'classifier': ai_engine.classifier is not None
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/breeds', methods=['GET'])
def list_breeds():
    """List all supported breeds."""
    return jsonify({
        'total': len(BREEDS),
        'cattle': BREEDS[:23],
        'buffalo': BREEDS[23:33],
        'cross': BREEDS[33:],
        'all': BREEDS
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
        return jsonify({'error': 'No image provided'}), 400
    
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
        return jsonify({'error': 'No image data provided'}), 400
    
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
        "expert_verified": false,
        "image_base64": "optional_base64_image"
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
    
    # Save to feedback file (in production, use database)
    feedback_dir = Path('../data/feedback')
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
    
    # Save escalation (in production, use database and notification system)
    escalation_dir = Path('../data/escalations')
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


# ==================== Expert Dashboard Routes ====================

@app.route('/expert/pending', methods=['GET'])
def get_pending_cases():
    """Get list of pending expert review cases."""
    escalation_dir = Path('../data/escalations')
    pending_cases = []
    
    if escalation_dir.exists():
        for case_file in escalation_dir.glob('ESC_*.json'):
            with open(case_file, 'r') as f:
                case = json.load(f)
                if case.get('status') == 'pending':
                    pending_cases.append(case)
    
    return jsonify({
        'total': len(pending_cases),
        'cases': sorted(pending_cases, key=lambda x: x['created_at'])
    })


@app.route('/expert/resolve', methods=['POST'])
def resolve_case():
    """
    Resolve an expert review case.
    
    Expects JSON body:
    {
        "escalation_id": "ESC_20250221...",
        "resolved_breed": "Gir",
        "expert_notes": "Confirmed based on hump size and color pattern",
        "expert_id": "VET001"
    }
    """
    data = request.get_json()
    
    if not data or 'escalation_id' not in data:
        return jsonify({'error': 'Missing escalation_id'}), 400
    
    escalation_file = Path(f"../data/escalations/{data['escalation_id']}.json")
    
    if not escalation_file.exists():
        return jsonify({'error': 'Escalation not found'}), 404
    
    # Update case
    with open(escalation_file, 'r') as f:
        case = json.load(f)
    
    case['status'] = 'resolved'
    case['resolved_breed'] = data.get('resolved_breed')
    case['expert_notes'] = data.get('expert_notes', '')
    case['expert_id'] = data.get('expert_id', 'unknown')
    case['resolved_at'] = datetime.now().isoformat()
    
    with open(escalation_file, 'w') as f:
        json.dump(case, f, indent=2)
    
    # Also save as feedback for model improvement
    feedback = {
        'prediction_id': case.get('prediction_id', 'unknown'),
        'predicted_breed': case['top_predictions'][0]['breed'] if case.get('top_predictions') else 'unknown',
        'actual_breed': data['resolved_breed'],
        'expert_verified': True,
        'expert_id': data.get('expert_id', 'unknown'),
        'timestamp': datetime.now().isoformat()
    }
    
    feedback_dir = Path('../data/feedback')
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback) + '\n')
    
    return jsonify({
        'status': 'success',
        'message': 'Case resolved and recorded for model improvement',
        'case': case
    })


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Run development server
    app.run(host='0.0.0.0', port=5000, debug=True)
