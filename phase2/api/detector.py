"""
YOLO-based Bangladeshi Taka Detection Module.
Wraps the trained YOLOv12 model for inference on currency images.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    CLASS_NAMES,
    DENOMINATION_MAP
)


class TakaDetector:
    """
    Wrapper class for YOLOv12 model to detect Bangladeshi Taka currency notes.
    
    Attributes:
        model: Loaded YOLO model instance
        confidence_threshold: Minimum confidence for detections
        image_size: Input image size for the model
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        image_size: Optional[int] = None
    ):
        """
        Initialize the TakaDetector with model weights.
        
        Args:
            model_path: Path to the trained YOLO weights file (.pt)
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            image_size: Input image size for inference
        """
        self.model_path = model_path or MODEL_PATH
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.image_size = image_size or IMAGE_SIZE
        self.model: Optional[YOLO] = None
        self._is_loaded = False
    
    def load_model(self) -> None:
        """
        Load the YOLO model from the specified weights file.
        
        Raises:
            FileNotFoundError: If the model weights file doesn't exist
            RuntimeError: If the model fails to load
        """
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {model_path}. "
                "Please ensure the trained model is available."
            )
        
        try:
            self.model = YOLO(str(model_path))
            self._is_loaded = True
            print(f"✅ Model loaded successfully from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        confidence_threshold: Optional[float] = None,
        return_annotated: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on an input image and return detection results.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            confidence_threshold: Override default confidence threshold
            return_annotated: Whether to return annotated image data
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating successful prediction
                - detections: List of detection dictionaries
                - count: Number of detections
                - image_size: Original image dimensions
                - processing_time_ms: Inference time in milliseconds
                - annotated_image: (optional) Base64 encoded annotated image
                
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If image is invalid or cannot be processed
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() first."
            )
        
        conf_threshold = confidence_threshold or self.confidence_threshold
        
        # Start timing
        start_time = time.time()
        
        # Convert image to PIL if needed for size extraction
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        original_width, original_height = pil_image.size
        
        # Run YOLO inference
        results = self.model.predict(
            source=pil_image,
            conf=conf_threshold,
            imgsz=self.image_size,
            verbose=False
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Extract detections
        detections = []
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = box.xyxy[0].cpu().numpy()
                
                detection = {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                    "denomination": DENOMINATION_MAP.get(class_id, "Unknown"),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(float(coords[0]), 2),
                        "y1": round(float(coords[1]), 2),
                        "x2": round(float(coords[2]), 2),
                        "y2": round(float(coords[3]), 2)
                    }
                }
                detections.append(detection)
        
        # Build response
        response = {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": {
                "width": original_width,
                "height": original_height
            },
            "processing_time_ms": round(processing_time_ms, 2)
        }
        
        # Optionally add annotated image
        if return_annotated and len(detections) > 0:
            import base64
            from io import BytesIO
            
            annotated_array = result.plot()
            annotated_pil = Image.fromarray(annotated_array)
            
            buffer = BytesIO()
            annotated_pil.save(buffer, format="PNG")
            buffer.seek(0)
            
            response["annotated_image"] = base64.b64encode(buffer.read()).decode("utf-8")
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "denominations": list(DENOMINATION_MAP.values())
        }


# Global detector instance for API use
_detector: Optional[TakaDetector] = None


def get_detector() -> TakaDetector:
    """
    Get or create the global TakaDetector instance.
    
    Returns:
        Initialized TakaDetector instance
    """
    global _detector
    if _detector is None:
        _detector = TakaDetector()
    return _detector


def initialize_detector() -> TakaDetector:
    """
    Initialize and load the global detector.
    Called during API startup.
    
    Returns:
        Loaded TakaDetector instance
    """
    detector = get_detector()
    if not detector.is_loaded:
        detector.load_model()
    return detector
