"""
Pydantic schemas for API request/response validation.
Defines the data models for the Bangladeshi Taka Detection API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detection."""
    x1: float = Field(..., description="Left x-coordinate")
    y1: float = Field(..., description="Top y-coordinate")
    x2: float = Field(..., description="Right x-coordinate")
    y2: float = Field(..., description="Bottom y-coordinate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x1": 120.5,
                "y1": 85.3,
                "x2": 450.2,
                "y2": 320.8
            }
        }


class Detection(BaseModel):
    """Single detection result from the model."""
    class_id: int = Field(..., description="Numeric class identifier (0-8)")
    class_name: str = Field(..., description="Name of the detected currency class")
    denomination: str = Field(..., description="Currency denomination in Bengali format (e.g., ৳100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_id": 7,
                "class_name": "one hundred taka",
                "denomination": "৳100",
                "confidence": 0.9234,
                "bbox": {
                    "x1": 120.5,
                    "y1": 85.3,
                    "x2": 450.2,
                    "y2": 320.8
                }
            }
        }


class ImageSize(BaseModel):
    """Dimensions of the input image."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class PredictionResponse(BaseModel):
    """Successful prediction response from /predict endpoint."""
    success: bool = Field(True, description="Indicates successful processing")
    detections: List[Detection] = Field(default_factory=list, description="List of detected currency notes")
    count: int = Field(..., ge=0, description="Number of detections found")
    image_size: ImageSize = Field(..., description="Original image dimensions")
    processing_time_ms: float = Field(..., ge=0, description="Inference time in milliseconds")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image (if requested)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "detections": [
                    {
                        "class_id": 7,
                        "class_name": "one hundred taka",
                        "denomination": "৳100",
                        "confidence": 0.923,
                        "bbox": {"x1": 120.5, "y1": 85.3, "x2": 450.2, "y2": 320.8}
                    }
                ],
                "count": 1,
                "image_size": {"width": 640, "height": 480},
                "processing_time_ms": 45.2
            }
        }


class ErrorResponse(BaseModel):
    """Error response for failed requests."""
    success: bool = Field(False, description="Indicates failed processing")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Supported formats: JPEG, PNG"
            }
        }


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = Field(..., description="Health status of the API")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }


class APIInfoResponse(BaseModel):
    """Response for root endpoint with API information."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    model_info: Dict[str, Any] = Field(..., description="Model configuration details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Bangladeshi Taka Detection API",
                "version": "1.0.0",
                "description": "REST API for detecting Bangladeshi currency notes",
                "endpoints": {
                    "GET /": "API information",
                    "GET /health": "Health check",
                    "POST /predict": "Detect currency in image"
                },
                "model_info": {
                    "num_classes": 9,
                    "confidence_threshold": 0.25
                }
            }
        }
