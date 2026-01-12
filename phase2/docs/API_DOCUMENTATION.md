# Bangladeshi Taka Detection API - Documentation

## API Reference

### Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. GET / - API Information

Returns information about the API including available endpoints and model configuration.

**Request:**
```http
GET / HTTP/1.1
Host: localhost:8000
```

**Response (200 OK):**
```json
{
  "name": "Bangladeshi Taka Detection API",
  "version": "1.0.0",
  "description": "REST API for detecting Bangladeshi currency notes using YOLOv12",
  "endpoints": {
    "GET /": "API information",
    "GET /health": "Health check",
    "POST /predict": "Detect currency in uploaded image"
  },
  "model_info": {
    "model_path": "/app/model_weights/best.pt",
    "is_loaded": true,
    "confidence_threshold": 0.25,
    "image_size": 640,
    "num_classes": 9,
    "class_names": ["500 taka", "Fifty taka", "Five Taka", "One Taka", "One Thousand taka", "Ten Taka", "Twenty", "one hundred taka", "two taka"],
    "denominations": ["৳500", "৳50", "৳5", "৳1", "৳1000", "৳10", "৳20", "৳100", "৳2"]
  }
}
```

---

### 2. GET /health - Health Check

Returns the health status of the API and model.

**Request:**
```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Possible Status Values:**
- `healthy`: API is fully operational with model loaded
- `degraded`: API is running but model is not loaded

---

### 3. POST /predict - Currency Detection

Upload an image to detect Bangladeshi Taka currency notes.

**Request:**
```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file: <binary image data>
```

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|:----------|:-----|:---------|:--------|:------------|
| `confidence` | float | No | 0.25 | Confidence threshold (0.0-1.0) |
| `return_annotated` | boolean | No | false | Include annotated image in response |

**Request Examples:**

**curl:**
```bash
# Basic request
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@currency_image.jpg"

# With parameters
curl -X POST "http://localhost:8000/predict?confidence=0.5&return_annotated=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@currency_image.jpg"
```

**Python (requests):**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("currency_image.jpg", "rb")}
params = {"confidence": 0.5}

response = requests.post(url, files=files, params=params)
print(response.json())
```

**JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict?confidence=0.5', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response (200 OK):**
```json
{
  "success": true,
  "detections": [
    {
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
    },
    {
      "class_id": 1,
      "class_name": "Fifty taka",
      "denomination": "৳50",
      "confidence": 0.8756,
      "bbox": {
        "x1": 480.1,
        "y1": 92.7,
        "x2": 720.3,
        "y2": 315.4
      }
    }
  ],
  "count": 2,
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "processing_time_ms": 45.23,
  "annotated_image": null
}
```

**Response Fields:**
| Field | Type | Description |
|:------|:-----|:------------|
| `success` | boolean | Whether the request was processed successfully |
| `detections` | array | List of detected currency notes |
| `count` | integer | Number of detections |
| `image_size` | object | Original image dimensions |
| `processing_time_ms` | float | Time taken for inference in milliseconds |
| `annotated_image` | string/null | Base64 encoded annotated image (if requested) |

**Detection Object:**
| Field | Type | Description |
|:------|:-----|:------------|
| `class_id` | integer | Numeric class identifier (0-8) |
| `class_name` | string | Human-readable class name |
| `denomination` | string | Currency denomination with symbol |
| `confidence` | float | Detection confidence score (0.0-1.0) |
| `bbox` | object | Bounding box coordinates |

**Bounding Box Object:**
| Field | Type | Description |
|:------|:-----|:------------|
| `x1` | float | Left x-coordinate |
| `y1` | float | Top y-coordinate |
| `x2` | float | Right x-coordinate |
| `y2` | float | Bottom y-coordinate |

---

## Error Responses

### 400 Bad Request
Invalid input file or format.

```json
{
  "success": false,
  "error": "Invalid image format: text/plain. Supported: JPEG, PNG",
  "detail": null
}
```

### 422 Unprocessable Entity
Validation error on request parameters.

```json
{
  "detail": [
    {
      "loc": ["query", "confidence"],
      "msg": "ensure this value is less than or equal to 1",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 500 Internal Server Error
Server-side error during processing.

```json
{
  "success": false,
  "error": "Prediction failed: ...",
  "detail": null
}
```

### 503 Service Unavailable
Model not loaded or API not ready.

```json
{
  "success": false,
  "error": "Model not loaded. Please try again later.",
  "detail": null
}
```

---

## Currency Classes

| ID | Class Name | Denomination | Description |
|:--:|:-----------|:-------------|:------------|
| 0 | 500 taka | ৳500 | Five Hundred Taka note |
| 1 | Fifty taka | ৳50 | Fifty Taka note |
| 2 | Five Taka | ৳5 | Five Taka note |
| 3 | One Taka | ৳1 | One Taka note |
| 4 | One Thousand taka | ৳1000 | One Thousand Taka note |
| 5 | Ten Taka | ৳10 | Ten Taka note |
| 6 | Twenty | ৳20 | Twenty Taka note |
| 7 | one hundred taka | ৳100 | One Hundred Taka note |
| 8 | two taka | ৳2 | Two Taka note |

---

## Rate Limits

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting at the reverse proxy level (nginx, traefik, etc.).

---

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json
