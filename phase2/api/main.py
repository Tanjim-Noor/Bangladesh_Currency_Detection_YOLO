"""
Bangladeshi Taka Detection REST API
FastAPI application for currency note detection using YOLOv12.

Endpoints:
    GET  /         - API information and available endpoints
    GET  /health   - Health check for container probes
    POST /predict  - Detect currency notes in uploaded image
"""

import io
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    SUPPORTED_FORMATS,
    MAX_IMAGE_SIZE_MB
)
from .detector import get_detector, initialize_detector
from .schemas import (
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
    APIInfoResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the model on startup to avoid cold start delays.
    """
    # Startup: Load the YOLO model
    print("🚀 Starting Bangladeshi Taka Detection API...")
    try:
        initialize_detector()
        print("✅ API ready to serve requests")
    except Exception as e:
        print(f"⚠️ Warning: Model loading failed: {e}")
        print("   API will start but /predict will fail until model is available")
    
    yield
    
    # Shutdown: Cleanup if needed
    print("👋 Shutting down API...")


# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


@app.get(
    "/",
    response_model=APIInfoResponse,
    summary="API Information",
    description="Get information about the API and available endpoints"
)
async def root():
    """
    Root endpoint providing API information and available endpoints.
    """
    detector = get_detector()
    
    return APIInfoResponse(
        name=API_TITLE,
        version=API_VERSION,
        description="REST API for detecting Bangladeshi currency notes using YOLOv12",
        endpoints={
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Detect currency in uploaded image"
        },
        model_info=detector.get_model_info()
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health status and model availability"
)
async def health_check():
    """
    Health check endpoint for container orchestration and monitoring.
    Returns the status of the API and whether the model is loaded.
    """
    detector = get_detector()
    
    return HealthResponse(
        status="healthy" if detector.is_loaded else "degraded",
        model_loaded=detector.is_loaded
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Invalid image or request"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    },
    summary="Detect Currency",
    description="Upload an image to detect Bangladeshi Taka currency notes"
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG or PNG)"),
    confidence: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (0.0-1.0). Uses default if not specified."
    ),
    return_annotated: bool = Query(
        default=False,
        description="Return base64 encoded annotated image with detections"
    )
):
    """
    Detect Bangladeshi Taka currency notes in an uploaded image.
    
    - **file**: Image file in JPEG or PNG format
    - **confidence**: Optional confidence threshold (default: 0.25)
    - **return_annotated**: Whether to return annotated image with detections
    
    Returns detected currency denominations with confidence scores and bounding boxes.
    """
    detector = get_detector()
    
    # Check if model is loaded
    if not detector.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate content type
    content_type = file.content_type
    if content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {content_type}. Supported: JPEG, PNG"
        )
    
    # Read and validate file
    try:
        contents = await file.read()
        
        # Check file size
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.1f}MB. Maximum: {MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Validate image
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify it's a valid image
        
        # Reopen after verify (verify() moves the pointer)
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(e)}"
        )
    
    # Run prediction
    try:
        result = detector.predict(
            image=image,
            confidence_threshold=confidence,
            return_annotated=return_annotated
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom exception handler for consistent error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": None
        }
    )


# Entry point for running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT
    
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

