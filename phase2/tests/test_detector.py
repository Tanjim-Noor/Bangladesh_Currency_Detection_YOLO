"""
Unit Tests for the TakaDetector class.

Tests the YOLO model wrapper functionality including:
- Model loading
- Inference execution
- Result parsing
- Error handling
"""

import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

# Add phase2 to path
PHASE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2_DIR))

from api.detector import TakaDetector, get_detector, initialize_detector
from api.config import CLASS_NAMES, DENOMINATION_MAP


class TestTakaDetector:
    """Tests for TakaDetector class."""
    
    def test_detector_initialization(self):
        """Test that detector initializes with default values."""
        detector = TakaDetector()
        
        assert detector.confidence_threshold == 0.25
        assert detector.image_size == 640
        assert detector.is_loaded is False
    
    def test_detector_custom_initialization(self):
        """Test detector initialization with custom values."""
        detector = TakaDetector(
            confidence_threshold=0.5,
            image_size=320
        )
        
        assert detector.confidence_threshold == 0.5
        assert detector.image_size == 320
    
    def test_model_loading(self, detector: TakaDetector):
        """Test that model loads successfully."""
        assert detector.is_loaded is True
        assert detector.model is not None
    
    def test_model_info(self, detector: TakaDetector):
        """Test get_model_info returns correct structure."""
        info = detector.get_model_info()
        
        assert "model_path" in info
        assert "is_loaded" in info
        assert "confidence_threshold" in info
        assert "image_size" in info
        assert "num_classes" in info
        assert "class_names" in info
        assert "denominations" in info
        
        assert info["num_classes"] == 9
        assert len(info["class_names"]) == 9
    
    def test_predict_with_pil_image(self, detector: TakaDetector):
        """Test prediction with PIL Image input."""
        # Create a test image
        img = Image.new("RGB", (640, 640), color="white")
        
        result = detector.predict(img)
        
        assert "success" in result
        assert "detections" in result
        assert "count" in result
        assert "image_size" in result
        assert "processing_time_ms" in result
        
        assert result["success"] is True
        assert result["image_size"]["width"] == 640
        assert result["image_size"]["height"] == 640
    
    def test_predict_with_numpy_array(self, detector: TakaDetector):
        """Test prediction with numpy array input."""
        img_array = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        result = detector.predict(img_array)
        
        assert result["success"] is True
        assert result["image_size"]["width"] == 640
        assert result["image_size"]["height"] == 480
    
    def test_predict_with_real_image(self, detector: TakaDetector, sample_image_path: Path):
        """Test prediction with a real currency image."""
        result = detector.predict(str(sample_image_path))
        
        assert result["success"] is True
        assert isinstance(result["detections"], list)
        assert result["processing_time_ms"] > 0
    
    def test_predict_with_custom_confidence(self, detector: TakaDetector):
        """Test prediction with custom confidence threshold."""
        img = Image.new("RGB", (640, 640), color="white")
        
        result = detector.predict(img, confidence_threshold=0.8)
        
        assert result["success"] is True
    
    def test_predict_with_annotated_return(self, detector: TakaDetector, sample_image_path: Path):
        """Test prediction with annotated image return."""
        result = detector.predict(str(sample_image_path), return_annotated=True)
        
        assert result["success"] is True
        
        # If detections found, should have annotated image
        if result["count"] > 0:
            assert "annotated_image" in result
            assert len(result["annotated_image"]) > 0
    
    def test_detection_format(self, detector: TakaDetector, sample_image_path: Path):
        """Test that detections have correct format."""
        result = detector.predict(str(sample_image_path))
        
        if result["count"] > 0:
            detection = result["detections"][0]
            
            # Check all required fields
            assert "class_id" in detection
            assert "class_name" in detection
            assert "denomination" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            
            # Check types
            assert isinstance(detection["class_id"], int)
            assert isinstance(detection["class_name"], str)
            assert isinstance(detection["denomination"], str)
            assert isinstance(detection["confidence"], float)
            assert isinstance(detection["bbox"], dict)
            
            # Check confidence range
            assert 0.0 <= detection["confidence"] <= 1.0
            
            # Check class_id is valid
            assert 0 <= detection["class_id"] < 9
            
            # Check bbox has all coordinates
            bbox = detection["bbox"]
            assert all(key in bbox for key in ["x1", "y1", "x2", "y2"])


class TestGlobalDetector:
    """Tests for global detector functions."""
    
    def test_get_detector_returns_instance(self):
        """Test that get_detector returns a TakaDetector instance."""
        detector = get_detector()
        assert isinstance(detector, TakaDetector)
    
    def test_get_detector_returns_same_instance(self):
        """Test that get_detector returns the same instance."""
        detector1 = get_detector()
        detector2 = get_detector()
        assert detector1 is detector2
    
    def test_initialize_detector(self):
        """Test that initialize_detector loads the model."""
        detector = initialize_detector()
        assert detector.is_loaded is True


class TestClassMappings:
    """Tests for class name and denomination mappings."""
    
    def test_class_names_count(self):
        """Test that we have 9 class names."""
        assert len(CLASS_NAMES) == 9
    
    def test_denomination_map_count(self):
        """Test that we have 9 denominations."""
        assert len(DENOMINATION_MAP) == 9
    
    def test_denomination_map_keys(self):
        """Test that denomination map has keys 0-8."""
        for i in range(9):
            assert i in DENOMINATION_MAP
    
    def test_denomination_format(self):
        """Test that denominations have Bengali Taka symbol."""
        for denomination in DENOMINATION_MAP.values():
            assert denomination.startswith("৳")
