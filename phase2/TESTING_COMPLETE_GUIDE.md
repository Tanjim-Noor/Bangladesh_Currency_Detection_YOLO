# API Testing & Validation - Complete Step-by-Step Guide

## Overview
This guide provides complete instructions to test the Bangladeshi Taka Detection API running in Docker. You'll run commands, capture screenshots, and document results for submission.

---

## STEP 1: Verify Docker Container is Running

> 💡 **Windows PowerShell note:** If you're running these commands in PowerShell, use `curl.exe` (instead of `curl`) to avoid PowerShell's `curl` alias interfering with flags like `-X` and `-F`. Example: `curl.exe -X POST -F "file=@path/to/image.jpg" "http://localhost:8000/predict"`.

### Command:
```bash
docker-compose -f phase2/docker/docker-compose.yml ps
```

### What to Look For:
- Container Name: `bd-taka-detector`
- Status: Should show `Up ... (healthy)`
- Port: Should show `0.0.0.0:8000->8000/tcp`

### Screenshot To Take:
Take a screenshot showing the container is running and healthy.

---

## STEP 2: Test Health Endpoint (Health Check)

### Command:
```bash
curl http://localhost:8000/health
```

### Expected Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### What to Verify:
- [OK] Status is "healthy"
- [OK] Model is loaded (true)
- [OK] Response time should be <100ms

### Screenshot To Take:
Capture the terminal showing the health check response.

---

## STEP 3: Test Root Endpoint (API Metadata)

### Command:
```bash
curl http://localhost:8000/
```

### Expected Response:
```json
{
  "name": "Bangladeshi Taka Detection API",
  "version": "1.0.0",
  "description": "REST API for detecting Bangladeshi currency notes using YOLOv12",
  "endpoints": {...},
  "model_info": {...}
}
```

### What to Verify:
- [OK] API name and version are correct
- [OK] All endpoints are listed
- [OK] Model info shows 9 classes

### Screenshot To Take:
Capture showing API metadata returned.

---

## STEP 4-9: Test with 6 Different Test Images

### Available Test Images:

| Test # | Image | Denomination | Command |
|--------|-------|--------------|---------|
| 4 | 1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg | 1000 Taka | See below |
| 5 | 50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg | 50 Taka | See below |
| 6 | 100_5_jpg.rf.41a66e15c259696166e6d943dab6a84e.jpg | 100 Taka | See below |
| 7 | 500_2_jpg.rf.9537954b6dddfaf25935c2483322ce1f.jpg | 500 Taka | See below |
| 8 | 5_6_jpg.rf.519666840f96377378e52073ee7eea9a.jpg | 5 Taka | See below |
| 9 | 10_12_jpg.rf.77efb45bc8a1826ab805fd6112b23fdb.jpg | 10 Taka | See below |

---

### TEST 4: 1000 Taka Notes

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- `"success": true`
- `"count"`: Should detect multiple 1000 Taka notes
- `"class_name"`: Should show "One Thousand taka"
- `"denomination"`: Should show "৳1000"
- `"confidence"`: Values between 0 and 1 (look for >0.7)
- `"processing_time_ms"`: First run ~2800ms, note it

**Screenshot To Take:**
Capture the complete JSON response showing detections.

---

### TEST 5: 50 Taka Notes

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- `"count"`: Number of 50 Taka notes detected
- `"class_name"`: Should show "Fifty taka"
- Confidence scores
- Bounding box coordinates (should be within image dimensions)

**Screenshot To Take:**
Capture the response showing 50 Taka detections.

---

### TEST 6: 100 Taka Notes

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/100_5_jpg.rf.41a66e15c259696166e6d943dab6a84e.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- Detections for 100 Taka notes
- Processing time (should be faster, ~40-100ms)
- Accuracy of denomination classification

**Screenshot To Take:**
Capture showing 100 Taka predictions.

---

### TEST 7: 500 Taka Notes

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/500_2_jpg.rf.9537954b6dddfaf25935c2483322ce1f.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- Single or multiple 500 Taka note detections
- `"class_name"`: Should show "500 taka"
- Confidence scores

**Screenshot To Take:**
Capture 500 Taka results.

---

### TEST 8: 5 Taka Notes (Small Denomination)

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/5_6_jpg.rf.519666840f96377378e52073ee7eea9a.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- Smallest denomination detection
- `"class_name"`: Should show "Five Taka"
- Confidence and accuracy

**Screenshot To Take:**
Capture 5 Taka detection results.

---

### TEST 9: 10 Taka Notes

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/10_12_jpg.rf.77efb45bc8a1826ab805fd6112b23fdb.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool
```

**What to Look For:**
- 10 Taka note detections
- `"class_name"`: Should show "Ten Taka"
- Bounding boxes and confidence

**Screenshot To Take:**
Capture 10 Taka results.

---

## STEP 10: Test with Annotated Image

### Command:
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" "http://localhost:8000/predict?confidence_threshold=0.25&annotated_image=true" | python -m json.tool
```

### What to Look For:
- Same detection data as before
- `"annotated_image"`: Will contain base64 encoded image with bounding boxes
- Response will be longer due to image data

### Screenshot To Take:
Capture showing the annotated_image field (or first part of response).

---

## STEP 11: Test Error Handling

### Test 11A: Invalid Confidence Threshold

**Command:**
```bash
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg" "http://localhost:8000/predict?confidence_threshold=1.5"
```

**Expected:** Error response for invalid threshold

**Screenshot To Take:**
Capture error response.

---

### Test 11B: Missing File

**Command:**
```bash
curl.exe -X POST -F "file=@nonexistent.jpg" "http://localhost:8000/predict"
```

**Expected:** Error about file not found

**Screenshot To Take:**
Capture error handling.

---

## STEP 12: Access Swagger UI for Interactive Testing

### URL:
```
http://localhost:8000/docs
```

### Steps:
1. Open your browser
2. Navigate to http://localhost:8000/docs
3. Click on "POST /predict"
4. Click "Try it out"
5. Upload an image and click "Execute"
6. View the response

### Screenshot To Take:
- Screenshot of Swagger UI page
- Screenshot showing prediction results in Swagger

---

## STEP 13: Performance Metrics Collection

### Run All Tests and Record Times

For each test image, note:

| Metric | Value |
|--------|-------|
| Image | 1000_0_jpg |
| Denomination | 1000 Taka |
| Detections | 3 |
| Processing Time (ms) | ~2891 |
| Min Confidence | 0.7314 |
| Max Confidence | 0.8436 |
| Avg Confidence | 0.8382 |

**Screenshot To Take:**
Create a table or document showing all metrics.

---

## 📝 Test Results Documentation Template

Create a document with the following structure:

```
# API Testing Results

## Test Environment
- Docker Container: bd-taka-detector
- API Version: 1.0.0
- Model: YOLOv12 (Phase 1)
- Port: 8000

## Test 1: Health Check
- Status: [OK] PASS
- Response: {"status": "healthy", "model_loaded": true}

## Test 2: Root Endpoint
- Status: [OK] PASS
- Returns: API metadata with 9 classes

## Test 3-8: Image Predictions
### Test 3: 1000 Taka
- Input Image: 1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg
- Detections: 3 notes
- Classes: One Thousand taka
- Confidence: 0.8436, 0.8314, 0.7314
- Processing Time: 2891.32 ms
- Status: [OK] PASS

### Test 4: 50 Taka
...

## Accuracy Discussion

### Detection Accuracy: 95%
- Correctly identified all visible notes
- No false positives observed
- Average confidence: 0.82

### Classification Accuracy: 100%
- All detected denominations matched actual values
- No misclassifications

### Performance:
- First prediction: ~2.9 seconds
- Subsequent: ~50-100 ms
- Acceptable for production

```

---

## 🔍 Expected Response Format Reference

### Successful Prediction:
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
  "image_size": {"width": 416, "height": 416},
  "processing_time_ms": 2891.32,
  "annotated_image": null
}
```

### Error Response:
```json
{
  "success": false,
  "error": "Invalid image format",
  "detail": "Supported formats: JPEG, PNG, BMP, WebP"
}
```

---

## 📸 Screenshots Checklist

Required screenshots:
- [ ] Container running status
- [ ] Health endpoint response
- [ ] Root endpoint response
- [ ] Test 1: 1000 Taka predictions
- [ ] Test 2: 50 Taka predictions
- [ ] Test 3: 100 Taka predictions
- [ ] Test 4: 500 Taka predictions
- [ ] Test 5: 5 Taka predictions
- [ ] Test 6: 10 Taka predictions
- [ ] Test 7: Annotated image response
- [ ] Test 8: Error handling
- [ ] Swagger UI interface
- [ ] Swagger UI prediction test

---

## 📊 Accuracy Analysis Template

### Detection Accuracy:
- What percentage of visible notes were detected?
- Were there any false positives?
- What was the average confidence?

### Classification Accuracy:
- Were all detected denominations correct?
- Any misclassifications?
- Which denominations had highest confidence?

### Bounding Box Accuracy:
- Were boxes properly fitted around notes?
- Did any boxes go outside image bounds?
- Quality of box placement (tight vs loose)?

### Performance:
- First prediction time (includes model warmup): ~2.9 seconds
- Subsequent predictions: ~40-100 ms
- Is this acceptable?
- Production viability?

### Edge Cases:
- Partially visible notes: Detected?
- Overlapping notes: Handled correctly?
- Different angles/rotations: Performance?
- Low quality images: Performance?

---

## Quick Command Reference

Copy these commands to easily run tests:

```bash
# Health check
curl.exe http://localhost:8000/health

# Root endpoint
curl.exe http://localhost:8000/

# Test 1: 1000 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/1000_0_jpg.rf.ab81c647da461f1a584b3f63aad96455.jpg" "http://localhost:8000/predict?confidence_threshold=0.25" | python -m json.tool

# Test 2: 50 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" "http://localhost:8000/predict" | python -m json.tool

# Test 3: 100 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/100_5_jpg.rf.41a66e15c259696166e6d943dab6a84e.jpg" "http://localhost:8000/predict" | python -m json.tool

# Test 4: 500 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/500_2_jpg.rf.9537954b6dddfaf25935c2483322ce1f.jpg" "http://localhost:8000/predict" | python -m json.tool

# Test 5: 5 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/5_6_jpg.rf.519666840f96377378e52073ee7eea9a.jpg" "http://localhost:8000/predict" | python -m json.tool

# Test 6: 10 Taka
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/10_12_jpg.rf.77efb45bc8a1826ab805fd6112b23fdb.jpg" "http://localhost:8000/predict" | python -m json.tool

# Test with annotation
curl.exe -X POST -F "file=@phase1/dataset/filtered/test/images/50_1_jpg.rf.d9a68e13a5991deb9fc8aa8b40ebb6f3.jpg" "http://localhost:8000/predict?annotated_image=true" | python -m json.tool

# Open Swagger UI
# http://localhost:8000/docs
```

---

## 📦 Deliverables to Submit

1. **Test Images**: Save copies of the 6 test images used
2. **Screenshots**: 12-13 screenshots showing all tests
3. **API Responses**: JSON responses from each test
4. **Analysis Document**: Accuracy discussion and metrics
5. **Test Report**: Complete documentation of all tests

---

## Success Criteria

Your tests should demonstrate:
- [OK] API is accessible and running
- [OK] All endpoints respond correctly
- [OK] Predictions are accurate (correct denominations)
- [OK] Response format is correct
- [OK] Multiple test images with different denominations
- [OK] Error handling works
- [OK] Performance is acceptable
- [OK] Screenshots and documentation are complete

---

**Good luck with your testing! Capture clear screenshots and document your findings.**
