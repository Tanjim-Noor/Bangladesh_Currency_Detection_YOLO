"""
Configuration settings for the Bangladeshi Taka Detection API.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_WEIGHTS_DIR / "best.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Bangladeshi Taka Detection API"
API_DESCRIPTION = """
REST API for detecting Bangladeshi currency notes using YOLOv12.

## Features
- Detect 9 different Taka denominations from images
- Returns class names, confidence scores, and bounding box coordinates
- Supports JPEG and PNG image formats

## Denominations Detected
- ৳1 (One Taka)
- ৳2 (Two Taka)
- ৳5 (Five Taka)
- ৳10 (Ten Taka)
- ৳20 (Twenty Taka)
- ৳50 (Fifty Taka)
- ৳100 (One Hundred Taka)
- ৳500 (Five Hundred Taka)
- ৳1000 (One Thousand Taka)
"""
API_VERSION = "1.0.0"

# Currency class mappings (from Phase 1 training)
CLASS_NAMES = [
    "500 taka",           # 0 - ৳500
    "Fifty taka",         # 1 - ৳50
    "Five Taka",          # 2 - ৳5
    "One Taka",           # 3 - ৳1
    "One Thousand taka",  # 4 - ৳1000
    "Ten Taka",           # 5 - ৳10
    "Twenty",             # 6 - ৳20
    "one hundred taka",   # 7 - ৳100
    "two taka"            # 8 - ৳2
]

DENOMINATION_MAP = {
    0: "৳500",
    1: "৳50",
    2: "৳5",
    3: "৳1",
    4: "৳1000",
    5: "৳10",
    6: "৳20",
    7: "৳100",
    8: "৳2"
}

# Supported image formats
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
