"""
API Endpoint Tests for Bangladeshi Taka Detection API.

This module contains comprehensive tests for all API endpoints:
- GET /        : API information endpoint
- GET /health  : Health check endpoint
- POST /predict: Currency detection endpoint

Usage:
    pytest phase2/tests/test_api.py -v
    pytest phase2/tests/test_api.py -v --tb=short
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import get_test_image_files, PHASE1_TEST_IMAGES


# =============================================================================
# Root Endpoint Tests
# =============================================================================

class TestRootEndpoint:
    """Tests for the GET / endpoint."""
    
    def test_root_returns_200(self, test_client: TestClient):
        """Test that root endpoint returns 200 OK."""
        response = test_client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self, test_client: TestClient):
        """Test that root endpoint returns API information."""
        response = test_client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "model_info" in data
    
    def test_root_contains_correct_endpoints(self, test_client: TestClient):
        """Test that root endpoint lists all available endpoints."""
        response = test_client.get("/")
        endpoints = response.json()["endpoints"]
        
        assert "GET /" in endpoints
        assert "GET /health" in endpoints
        assert "POST /predict" in endpoints


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""
    
    def test_health_returns_200(self, test_client: TestClient):
        """Test that health endpoint returns 200 OK."""
        response = test_client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, test_client: TestClient):
        """Test that health endpoint returns status field."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
    
    def test_health_returns_model_loaded_status(self, test_client: TestClient):
        """Test that health endpoint returns model_loaded field."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


# =============================================================================
# Predict Endpoint Tests
# =============================================================================

class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""
    
    def test_predict_with_valid_image(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test prediction with a valid image returns 200."""
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
    
    def test_predict_returns_correct_structure(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test that prediction response has correct structure."""
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()
        
        assert "success" in data
        assert "detections" in data
        assert "count" in data
        assert "image_size" in data
        assert "processing_time_ms" in data
        
        assert data["success"] is True
        assert isinstance(data["detections"], list)
        assert isinstance(data["count"], int)
        assert data["count"] >= 0
    
    def test_predict_detection_structure(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test that each detection has correct fields."""
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()
        
        # If there are detections, verify their structure
        if data["count"] > 0:
            detection = data["detections"][0]
            
            assert "class_id" in detection
            assert "class_name" in detection
            assert "denomination" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            
            # Verify bbox structure
            bbox = detection["bbox"]
            assert "x1" in bbox
            assert "y1" in bbox
            assert "x2" in bbox
            assert "y2" in bbox
    
    def test_predict_with_confidence_parameter(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test prediction with custom confidence threshold."""
        response = test_client.post(
            "/predict?confidence=0.5",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
    
    def test_predict_with_annotated_image(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test prediction with annotated image return."""
        response = test_client.post(
            "/predict?return_annotated=true",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        # If detections exist, annotated_image should be present
        if data["count"] > 0:
            assert "annotated_image" in data
    
    def test_predict_with_invalid_format(self, test_client: TestClient, invalid_file_content: bytes):
        """Test prediction with invalid file format returns 400."""
        response = test_client.post(
            "/predict",
            files={"file": ("test.txt", invalid_file_content, "text/plain")}
        )
        assert response.status_code == 400
    
    def test_predict_without_file(self, test_client: TestClient):
        """Test prediction without file returns 422."""
        response = test_client.post("/predict")
        assert response.status_code == 422
    
    def test_predict_with_png_image(self, test_client: TestClient):
        """Test prediction with PNG image."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        response = test_client.post(
            "/predict",
            files={"file": ("test.png", buffer.read(), "image/png")}
        )
        assert response.status_code == 200


# =============================================================================
# Multiple Image Tests (Validation)
# =============================================================================

class TestMultipleImages:
    """Tests with multiple images for validation."""
    
    def test_predict_with_multiple_images(self, test_client: TestClient, multiple_test_images: list[Path]):
        """Test prediction with multiple different images."""
        results = []
        
        for img_path in multiple_test_images:
            image_bytes = img_path.read_bytes()
            content_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
            
            response = test_client.post(
                "/predict",
                files={"file": (img_path.name, image_bytes, content_type)}
            )
            
            assert response.status_code == 200, f"Failed for image: {img_path.name}"
            
            data = response.json()
            results.append({
                "image": img_path.name,
                "count": data["count"],
                "detections": data["detections"],
                "processing_time_ms": data["processing_time_ms"]
            })
        
        # Log results summary
        print("\n" + "=" * 60)
        print("MULTIPLE IMAGE TEST RESULTS")
        print("=" * 60)
        
        for result in results:
            print(f"\nImage: {result['image']}")
            print(f"  Detections: {result['count']}")
            print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
            
            for det in result["detections"]:
                print(f"    - {det['class_name']} ({det['denomination']}): {det['confidence']:.2%}")
        
        print("\n" + "=" * 60)
    
    def test_prediction_accuracy_observations(self, test_client: TestClient, multiple_test_images: list[Path]):
        """
        Test and document prediction accuracy observations.
        This test provides insights into model performance.
        """
        total_images = len(multiple_test_images)
        images_with_detections = 0
        total_detections = 0
        confidence_scores = []
        denominations_found = set()
        
        for img_path in multiple_test_images:
            image_bytes = img_path.read_bytes()
            content_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
            
            response = test_client.post(
                "/predict",
                files={"file": (img_path.name, image_bytes, content_type)}
            )
            
            data = response.json()
            
            if data["count"] > 0:
                images_with_detections += 1
                total_detections += data["count"]
                
                for det in data["detections"]:
                    confidence_scores.append(det["confidence"])
                    denominations_found.add(det["denomination"])
        
        # Calculate metrics
        detection_rate = images_with_detections / total_images * 100 if total_images > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        print("\n" + "=" * 60)
        print("PREDICTION ACCURACY OBSERVATIONS")
        print("=" * 60)
        print(f"Total images tested: {total_images}")
        print(f"Images with detections: {images_with_detections} ({detection_rate:.1f}%)")
        print(f"Total detections: {total_detections}")
        print(f"Average confidence: {avg_confidence:.2%}")
        print(f"Unique denominations found: {', '.join(sorted(denominations_found))}")
        print("=" * 60 + "\n")
        
        # Assert basic expectations
        assert total_images >= 5, "Expected at least 5 test images"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_confidence_parameter(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test that invalid confidence parameter is rejected."""
        response = test_client.post(
            "/predict?confidence=1.5",  # Invalid: > 1.0
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 422
    
    def test_negative_confidence_parameter(self, test_client: TestClient, sample_image_bytes: bytes):
        """Test that negative confidence parameter is rejected."""
        response = test_client.post(
            "/predict?confidence=-0.5",  # Invalid: < 0.0
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 422
    
    def test_corrupted_image(self, test_client: TestClient):
        """Test handling of corrupted image data."""
        # Create bytes that look like an image header but are corrupted
        corrupted_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Partial JPEG header
        
        response = test_client.post(
            "/predict",
            files={"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
        )
        assert response.status_code == 400

