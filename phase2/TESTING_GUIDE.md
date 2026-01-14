# API Testing & Validation - Comprehensive Guide

## Overview
This document provides step-by-step instructions to test the Bangladeshi Taka Detection API running in Docker.

---

## Part 1: Docker Container Setup (Already Completed ✅)

### Verify Container Status
```bash
docker-compose -f phase2/docker/docker-compose.yml ps
```

**Expected Output:**
- Container Status: `Up and healthy`
- Port Mapping: `0.0.0.0:8000->8000/tcp`
- Health Check: Passing

### Check API Accessibility
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Part 2: API Testing with curl (Recommended for Automation)

### Test 1: Root Endpoint
```bash
curl http://localhost:8000/
```

**Purpose:** Verify API metadata and available endpoints

### Test 2: Single Image Prediction
```bash
curl -X POST \
  -F "file=@phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg" \
  "http://localhost:8000/predict?confidence_threshold=0.25" \
  | python -m json.tool
```

**Purpose:** Test basic prediction functionality

### Test 3: Prediction with Annotated Image
```bash
curl -X POST \
  -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" \
  "http://localhost:8000/predict?confidence_threshold=0.25&annotated_image=true" \
  | python -m json.tool
```

**Purpose:** Test annotated image generation

### Test 4: Error Handling - Invalid File
```bash
curl -X POST \
  -F "file=@nonexistent.jpg" \
  "http://localhost:8000/predict" 2>&1
```

**Purpose:** Verify error handling for missing files

### Test 5: Error Handling - Invalid Image Format
```bash
echo "This is not an image" > temp.txt
curl -X POST \
  -F "file=@temp.txt" \
  "http://localhost:8000/predict"
```

**Purpose:** Verify error handling for corrupted/invalid images

---

## Part 3: API Testing with Postman

### Steps to Set Up Postman:

1. **Download Postman** (if not installed)
   - Visit: https://www.postman.com/downloads/

2. **Import OpenAPI Documentation**
   - Open Postman
   - Click "Import" → Select "Link"
   - Enter: `http://localhost:8000/openapi.json`
   - Postman will auto-generate all endpoints

3. **Or Manually Create Requests:**

#### Request 1: GET /
- **Method:** GET
- **URL:** `http://localhost:8000/`
- **Headers:** None required
- **Body:** None

#### Request 2: POST /predict (Single Image)
- **Method:** POST
- **URL:** `http://localhost:8000/predict?confidence_threshold=0.25`
- **Headers:** None (Postman handles multipart/form-data)
- **Body:** 
  - Type: `form-data`
  - Key: `file`
  - Value: Select image file from filesystem
  - Params: `confidence_threshold=0.25`

#### Request 3: POST /predict (Multiple Images)
- Repeat Request 2 with different test images

---

## Part 4: Test Images Selection

### Recommended Test Images (at least 5):

| Test # | Image Name | Denomination | Purpose |
|--------|-----------|--------------|---------|
| 1 | `1000_0_jpg.rf.ab81c...jpg` | 1000 Taka | Multiple notes detection |
| 2 | `50_1_jpg.rf.d9a68e...jpg` | 50 Taka | Lower denomination |
| 3 | `100_5_jpg.rf.41a66e...jpg` | 100 Taka | Mid-range denomination |
| 4 | `500_2_jpg.rf.9537954...jpg` | 500 Taka | Higher denomination |
| 5 | `5_6_jpg.rf.519666...jpg` | 5 Taka | Smallest denomination |
| 6 | `10_12_jpg.rf.77efb45b...jpg` | 10 Taka | Additional validation |

**Location:** `phase1/dataset/filtered/test/images/`

---

## Part 5: Expected Response Format

### Successful Prediction Response

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 4,
      "class_name": "One Thousand taka",
      "denomination": "৳1000",
      "confidence": 0.8436,
      "bbox": {
        "x1": 286.29,
        "y1": 300.87,
        "x2": 378.92,
        "y2": 382.39
      }
    }
  ],
  "count": 3,
  "image_size": {
    "width": 416,
    "height": 416
  },
  "processing_time_ms": 2891.32,
  "annotated_image": null
}
```

### Error Response

```json
{
  "success": false,
  "error": "Invalid image format",
  "detail": "Supported formats: JPEG, PNG, BMP, WebP"
}
```

---

## Part 6: Validation Checklist

### Endpoint Functionality
- [ ] GET / returns API metadata
- [ ] GET /health returns status: "healthy"
- [ ] POST /predict accepts image files
- [ ] POST /predict returns predictions with correct structure
- [ ] Confidence threshold parameter works
- [ ] Annotated image generation works

### Response Format
- [ ] All responses include "success" field
- [ ] Detection objects include all required fields (class_id, class_name, confidence, bbox)
- [ ] Bounding box coordinates are numeric
- [ ] Processing time is recorded
- [ ] Image size is correct

### Prediction Accuracy
- [ ] Detected classes match actual denominations
- [ ] Confidence scores are reasonable (0.0 to 1.0)
- [ ] Bounding boxes are within image bounds
- [ ] Multiple objects detected correctly in single image

### Error Handling
- [ ] Invalid files return appropriate error
- [ ] Corrupted images handled gracefully
- [ ] Missing required parameters return error
- [ ] Invalid confidence thresholds handled

---

## Part 7: Performance Metrics to Record

For each test image, record:

| Metric | Value |
|--------|-------|
| Image Filename | - |
| Image Dimensions | WxH |
| Processing Time (ms) | - |
| Number of Detections | - |
| Average Confidence | - |
| Min/Max Confidence | - |

---

## Part 8: Discussion Points on Accuracy

### Evaluation Criteria:

1. **Detection Accuracy**
   - Are all visible currency notes detected?
   - Are false positives present?
   - What's the average confidence score?

2. **Classification Accuracy**
   - Are detected denominations correct?
   - Any misclassifications observed?

3. **Bounding Box Accuracy**
   - Are bounding boxes tightly fit around notes?
   - Are coordinates logical?

4. **Edge Cases**
   - Partially visible notes: Detected?
   - Overlapping notes: Handled correctly?
   - Different angles/rotations: Performance?

5. **Performance**
   - First prediction latency: ~2.9s (model warmup)
   - Subsequent predictions: ~40-100ms
   - Is this acceptable for production?

---

## Quick Reference: Common curl Commands

### Test Health Endpoint
```bash
curl http://localhost:8000/health | python -m json.tool
```

### List Test Images
```bash
ls phase1/dataset/filtered/test/images/*.jpg | head -10
```

### Test with Specific Image
```bash
IMAGE="phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg"
curl -X POST -F "file=@$IMAGE" "http://localhost:8000/predict" | python -m json.tool
```

### Test with Custom Confidence Threshold
```bash
curl -X POST \
  -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" \
  "http://localhost:8000/predict?confidence_threshold=0.5" \
  | python -m json.tool
```

### Save Response to File
```bash
curl -X POST \
  -F "file=@phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg" \
  "http://localhost:8000/predict" \
  > test_response_1.json

cat test_response_1.json | python -m json.tool
```

---

## Next: Execute Comprehensive Tests

See `TESTING_EXECUTION.md` for automated test scripts and detailed results.
