"""
Image Preprocessing Module
==========================
Handles image preprocessing for cattle breed recognition.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Preprocesses images for breed recognition model.
    
    Handles:
    - Image resizing
    - Normalization
    - Quality enhancement
    - Background handling
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Normalized image (float32)
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality.
        
        Applies:
        - Noise reduction
        - Contrast enhancement
        - Sharpness improvement
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        from PIL import ImageEnhance
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def check_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if image meets quality requirements.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (is_valid, message)
        """
        height, width = image.shape[:2]
        
        # Check minimum resolution
        if height < 300 or width < 300:
            return False, "Image resolution too low. Minimum 300x300 required."
        
        # Check if image is too dark
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 30:
            return False, "Image is too dark. Please capture in better lighting."
        
        # Check if image is too bright
        if brightness > 225:
            return False, "Image is too bright. Please avoid direct sunlight."
        
        # Check blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return False, "Image is blurry. Please hold camera steady."
        
        return True, "Image quality OK"
    
    def preprocess(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input image
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed image ready for model
        """
        # Check quality
        is_valid, message = self.check_image_quality(image)
        if not is_valid:
            print(f"Warning: {message}")
        
        # Enhance if requested
        if enhance:
            image = self.enhance_image(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Load image from file and preprocess.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.preprocess(image)


# Indian cattle breeds list
INDIAN_CATTLE_BREEDS = [
    # Milch Breeds
    "Gir",
    "Sahiwal", 
    "Red Sindhi",
    "Tharparkar",
    "Rathi",
    
    # Draught Breeds
    "Hallikar",
    "Amritmahal",
    "Khillari",
    "Kangayam",
    "Bargur",
    "Pulikulam",
    "Umblachery",
    
    # Dual Purpose Breeds
    "Hariana",
    "Kankrej",
    "Ongole",
    "Deoni",
    "Krishna Valley",
    "Dangi",
    
    # Hill Cattle
    "Punganur",
    "Vechur",
    "Malnad Gidda",
    
    # Crossbreeds
    "Jersey Cross",
    "HF Cross",
    "Jersey-Sahiwal Cross",
]

INDIAN_BUFFALO_BREEDS = [
    "Murrah",
    "Jaffrabadi",
    "Nili-Ravi",
    "Banni",
    "Pandharpuri",
    "Mehsana",
    "Surti",
    "Nagpuri",
    "Toda",
    "Bhadawari",
    "Jaffrabadi",
    "Kalahandi",
    "Chilika",
]

# All breeds combined
ALL_BREEDS = INDIAN_CATTLE_BREEDS + INDIAN_BUFFALO_BREEDS

# Breed to index mapping
BREED_TO_IDX = {breed: idx for idx, breed in enumerate(ALL_BREEDS)}
IDX_TO_BREED = {idx: breed for breed, idx in BREED_TO_IDX.items()}

NUM_CLASSES = len(ALL_BREEDS)
