"""
Pytest fixtures and configuration for Bangladeshi Taka Detection API tests.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add the phase2 directory to path for imports
PHASE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2_DIR))

from api.main import app
from api.detector import TakaDetector, get_detector


# =============================================================================
# Test Image Paths
# =============================================================================

# Path to test images directory
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

# Path to Phase 1 test images (fallback source)
PHASE1_TEST_IMAGES = PHASE2_DIR.parent / "phase1" / "dataset" / "filtered" / "test" / "images"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """
    Create a TestClient instance for API testing.
    The client is reused across all tests in the module for efficiency.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def detector() -> TakaDetector:
    """
    Create and load a TakaDetector instance.
    Loaded once per test session to avoid repeated model loading.
    """
    det = TakaDetector()
    det.load_model()
    return det


@pytest.fixture(scope="module")
def sample_image_path() -> Path:
    """
    Get a sample test image path for testing.
    Tries test_images directory first, falls back to Phase 1 test images.
    """
    # Try local test images first
    if TEST_IMAGES_DIR.exists():
        images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
        if images:
            return images[0]
    
    # Fallback to Phase 1 test images
    if PHASE1_TEST_IMAGES.exists():
        images = list(PHASE1_TEST_IMAGES.glob("*.jpg")) + list(PHASE1_TEST_IMAGES.glob("*.png"))
        if images:
            return images[0]
    
    pytest.skip("No test images available")


@pytest.fixture(scope="module")
def multiple_test_images() -> list[Path]:
    """
    Get multiple test image paths for comprehensive testing.
    Returns at least 5 images if available.
    """
    images = []
    
    # Collect from local test images
    if TEST_IMAGES_DIR.exists():
        images.extend(TEST_IMAGES_DIR.glob("*.jpg"))
        images.extend(TEST_IMAGES_DIR.glob("*.png"))
    
    # Collect from Phase 1 test images
    if PHASE1_TEST_IMAGES.exists():
        images.extend(PHASE1_TEST_IMAGES.glob("*.jpg"))
        images.extend(PHASE1_TEST_IMAGES.glob("*.png"))
    
    # Return up to 10 unique images
    unique_images = list(set(images))[:10]
    
    if len(unique_images) < 5:
        pytest.skip(f"Need at least 5 test images, found {len(unique_images)}")
    
    return unique_images[:5]


@pytest.fixture
def invalid_file_content() -> bytes:
    """Create invalid file content for error testing."""
    return b"This is not an image file"


@pytest.fixture
def sample_image_bytes(sample_image_path: Path) -> bytes:
    """Read sample image as bytes for file upload testing."""
    return sample_image_path.read_bytes()


# =============================================================================
# Utility Functions
# =============================================================================

def get_test_image_files(count: int = 5) -> list[tuple[str, bytes]]:
    """
    Get test image files as (filename, bytes) tuples for API testing.
    
    Args:
        count: Number of images to retrieve
        
    Returns:
        List of (filename, file_bytes) tuples
    """
    images = []
    
    # Try Phase 1 test images
    if PHASE1_TEST_IMAGES.exists():
        for img_path in list(PHASE1_TEST_IMAGES.glob("*.jpg"))[:count]:
            images.append((img_path.name, img_path.read_bytes()))
    
    return images
