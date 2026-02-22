"""
Models Module
=============
Contains model definitions for cattle breed recognition.
"""

from .yolo_detector import YOLODetector
from .efficientnet_classifier import EfficientNetClassifier

__all__ = ['YOLODetector', 'EfficientNetClassifier']
