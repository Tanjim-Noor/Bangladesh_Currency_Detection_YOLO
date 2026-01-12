# Phase 2: Bangladeshi Taka Detection - REST API & Deployment

## 📋 Phase 2 Overview

Phase 2 deploys the Phase 1 trained YOLOv12 model as a production-ready REST API with Docker containerization and comprehensive documentation. This phase transforms the training artifacts into a scalable, accessible service for detecting 9 Bangladeshi Taka currency denominations.

**Status:** ✅ IMPLEMENTED

---

## 🚀 Quick Start

### Option 1: Run Locally (Development)

```bash
# Navigate to project root
cd Bangladesh_Currency_Detection_YOLO

# Install dependencies
pip install -r requirements.txt

# Navigate to phase2 directory
cd phase2

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Run with Docker

```bash
# Navigate to phase2/docker directory
cd phase2/docker

# Build and start the container
docker-compose up --build

# Or build and run manually
docker build -t bd-taka-detector -f phase2/docker/Dockerfile .
docker run -p 8000:8000 bd-taka-detector
```

### Access the API

- **API Documentation (Swagger):** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Predict Endpoint:** POST http://localhost:8000/predict

---

## 📁 Phase 2 Folder Structure

```
phase2/
├── README_PHASE2.md                  # This file
│
├── model_weights/                    # Production model files
│   ├── best.pt                       # ⭐ Fine-tuned model (from Phase 1)
│   ├── last.pt                       # Last training checkpoint
│   ├── yolo12m.pt                    # Pre-trained baseline (medium)
│   └── yolo12n.pt                    # Pre-trained baseline (nano)
│
├── api/                              # REST API implementation
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # FastAPI application with endpoints
│   ├── schemas.py                    # Pydantic request/response models
│   ├── detector.py                   # YOLO inference wrapper class
│   └── config.py                     # Configuration and settings
│
├── docker/                           # Containerization
│   ├── Dockerfile                    # Multi-stage container build
│   ├── docker-compose.yml            # Service orchestration
│   └── .dockerignore                 # Build context exclusions
│
├── tests/                            # Test suite
│   ├── __init__.py                   # Test package initialization
│   ├── conftest.py                   # Pytest fixtures and configuration
│   ├── test_api.py                   # API endpoint tests
│   ├── test_detector.py              # Detector unit tests
│   └── test_images/                  # Sample test images
│
├── deployment/                       # Deployment documentation
│   └── ENV_TEMPLATE                  # Environment variables template
│
└── docs/                             # Additional documentation
    └── API_DOCUMENTATION.md          # Detailed API reference
```

---

## 🔌 API Endpoints

### GET / - API Information
Returns API metadata and available endpoints.

```bash
curl http://localhost:8000/
```

**Response:**
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
  "model_info": {...}
}
```

### GET /health - Health Check
Check API health status and model availability.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict - Currency Detection
Upload an image to detect Bangladeshi Taka currency notes.

**Parameters:**
- `file` (required): Image file (JPEG or PNG)
- `confidence` (optional): Confidence threshold (0.0-1.0, default: 0.25)
- `return_annotated` (optional): Return annotated image (default: false)

**curl Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Response:**
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
    }
  ],
  "count": 1,
  "image_size": {"width": 640, "height": 480},
  "processing_time_ms": 45.2
}
```

---

## 💵 Supported Denominations

| Class ID | Class Name | Denomination |
|:--------:|:-----------|:-------------|
| 0 | 500 taka | ৳500 |
| 1 | Fifty taka | ৳50 |
| 2 | Five Taka | ৳5 |
| 3 | One Taka | ৳1 |
| 4 | One Thousand taka | ৳1000 |
| 5 | Ten Taka | ৳10 |
| 6 | Twenty | ৳20 |
| 7 | one hundred taka | ৳100 |
| 8 | two taka | ৳2 |

---

## 🐳 Docker Commands

### Build the Image
```bash
# From project root
docker build -t bd-taka-detector -f phase2/docker/Dockerfile .
```

### Run the Container
```bash
# Basic run
docker run -p 8000:8000 bd-taka-detector

# Run with custom environment variables
docker run -p 8000:8000 \
  -e CONFIDENCE_THRESHOLD=0.3 \
  -e MAX_IMAGE_SIZE_MB=20 \
  bd-taka-detector

# Run in detached mode
docker run -d -p 8000:8000 --name taka-api bd-taka-detector
```

### Using Docker Compose
```bash
cd phase2/docker

# Build and start
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f bd-taka-api

# Stop
docker-compose down
```

---

## 🧪 Running Tests

```bash
# Navigate to phase2 directory
cd phase2

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ -v --cov=api

# Run tests with detailed output
pytest tests/ -v --tb=short
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `MODEL_PATH` | `model_weights/best.pt` | Path to YOLO model weights |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence threshold |
| `IMAGE_SIZE` | `640` | Input image size (pixels) |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum upload size |

See [deployment/ENV_TEMPLATE](deployment/ENV_TEMPLATE) for all available options.

---

## 📊 Error Handling

The API returns consistent error responses:

**400 Bad Request** - Invalid input
```json
{
  "success": false,
  "error": "Invalid image format: text/plain. Supported: JPEG, PNG",
  "detail": null
}
```

**422 Unprocessable Entity** - Validation error
```json
{
  "detail": [{"loc": ["query", "confidence"], "msg": "...", "type": "..."}]
}
```

**500 Internal Server Error** - Server error
```json
{
  "success": false,
  "error": "Prediction failed: ...",
  "detail": null
}
```

---

## 🔗 Related Resources

- **Phase 1 Documentation:** [../phase1/README_PHASE1.md](../phase1/README_PHASE1.md)
- **Training Notebook:** [../phase1/training/bangladeshi_taka_detection_yolov12.ipynb](../phase1/training/bangladeshi_taka_detection_yolov12.ipynb)
- **Project Index:** [../INDEX.md](../INDEX.md)
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Ultralytics YOLO:** https://docs.ultralytics.com/

│   ├── TROUBLESHOOTING.md            # Common issues (TO CREATE)
│   ├── ENV_TEMPLATE                  # Environment variables (TO CREATE)
│   ├── kubernetes/                   # K8s configs (optional)
│   │   └── deployment.yaml           # K8s manifest (TO CREATE)
│   └── nginx/                        # Reverse proxy config
│       └── nginx.conf                # Nginx configuration (TO CREATE)
│
└── docs/                             # Development documentation
    ├── SETUP.md                      # Development setup (TO CREATE)
    ├── ARCHITECTURE.md               # System architecture (TO CREATE)
    ├── API_DESIGN.md                 # API design decisions (TO CREATE)
    └── CONTRIBUTING.md               # Contribution guide (TO CREATE)
```

---

## 🎯 Phase 2 Objectives

### Primary Goals

1. **REST API Development**
   - [ ] Implement FastAPI application
   - [ ] Create inference endpoints
   - [ ] Add model loading/caching
   - [ ] Implement error handling
   - [ ] Add request validation

2. **Docker Containerization**
   - [ ] Create production Dockerfile
   - [ ] Setup docker-compose configuration
   - [ ] Configure volumes for model weights
   - [ ] Optimize image size
   - [ ] Test container build & run

3. **Testing & Quality**
   - [ ] Unit tests for detector module
   - [ ] Integration tests for API endpoints
   - [ ] Performance tests for inference
   - [ ] Test coverage reporting
   - [ ] Load testing

4. **Deployment Documentation**
   - [ ] API endpoint documentation
   - [ ] Deployment guide
   - [ ] Environment configuration
   - [ ] Troubleshooting guide
   - [ ] Architecture documentation

---

## 🔗 Phase 1 Integration Points

### Assets from Phase 1

**Model Weights:**
```
Source: ../phase1/training/bd_taka_detector/weights/best.pt
Purpose: ⭐ Main fine-tuned inference model for Phase 2 API
Size: ~40MB (YOLOv12 fine-tuned)
Location: ./model_weights/best.pt
```

**Dependencies:**
```
Source: ../requirements.txt (shared across entire project)
Purpose: Python packages for both Phase 1 and Phase 2
DO NOT create separate requirements.txt in phase2/
```

**Dataset Configuration:**
```yaml
Source: ../phase1/dataset/filtered/data.yaml
Purpose: Class names and mappings
Classes: 9 Bangladeshi currency denominations
Format: YAML
```

**Class Mappings:**
```
0: "500 taka"
1: "Fifty taka"
2: "Five Taka"
3: "One Taka"
4: "One Thousand taka"
5: "Ten Taka"
6: "Twenty"
7: "one hundred taka"
8: "two taka"
```

**Test Dataset:**
```
Source: ../phase1/dataset/filtered/test/images/
Purpose: API testing and validation
Count: 189 images
```

**Reference Notebooks:**
```
- ../phase1/training/bangladeshi_taka_detection_yolov12.ipynb
- ../phase1/training/gpu_diagnostics.ipynb
```

---

## 🏗️ Architecture Overview

### API Architecture

```
CLIENT REQUEST
     ↓
[API Gateway / Load Balancer]
     ↓
[FastAPI Application]
  ├─ Request Validation (Pydantic)
  ├─ Image Preprocessing
  ├─ Model Inference
  └─ Response Formatting
     ↓
[YOLO Model]
  └─ Phase 1 best.pt weights
     ↓
[Response with Detections]
     ↓
CLIENT RESPONSE
```

### Container Architecture

```
[Docker Container]
  ├─ Base Image: python:3.11-slim
  ├─ Dependencies: FastAPI, Ultralytics, etc.
  ├─ Model Weights: /app/models/best.pt
  ├─ API Code: /app/api/
  └─ Exposed Port: 8000
```

---

## 📋 Implementation Roadmap

### Phase 2A: Core API Development (Weeks 1-2)

**Week 1: Foundation**
- [ ] Setup FastAPI project structure
- [ ] Create data models (Pydantic)
- [ ] Implement detector wrapper
- [ ] Create basic endpoints

**Week 2: Enhancement**
- [ ] Add image preprocessing
- [ ] Implement caching
- [ ] Add error handling
- [ ] Create health check endpoint

### Phase 2B: Containerization (Weeks 2-3)

- [ ] Write Dockerfile
- [ ] Create docker-compose.yml
- [ ] Build and test container
- [ ] Optimize image size

### Phase 2C: Testing & Documentation (Weeks 3-4)

- [ ] Write unit tests
- [ ] Create integration tests
- [ ] Performance testing
- [ ] Complete documentation

### Phase 2D: Deployment (Weeks 4-5)

- [ ] Setup deployment environment
- [ ] Configure CI/CD pipeline
- [ ] Deploy to production
- [ ] Setup monitoring

---

## 📝 API Specifications (To Be Implemented)

### Endpoints Overview

#### 1. Health Check
```
GET /health
Purpose: API availability check
Response: {"status": "ok", "version": "1.0.0"}
```

#### 2. Model Info
```
GET /model/info
Purpose: Get model details
Response: {"model": "yolov12", "classes": 9, "framework": "ultralytics"}
```

#### 3. Detect Currency
```
POST /detect
Content-Type: multipart/form-data
Parameters: 
  - file: Image file (JPEG, PNG)
  - confidence: Detection confidence threshold (0.0-1.0, default=0.5)
Response: {
  "detections": [
    {
      "class": "500 taka",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "processing_time_ms": 45
}
```

#### 4. Batch Detect
```
POST /detect/batch
Content-Type: multipart/form-data
Parameters:
  - files: Multiple image files
  - confidence: Confidence threshold
Response: Array of detection results
```

#### 5. Model Configuration
```
GET /config
Purpose: Get runtime configuration
Response: {
  "model_path": "...",
  "input_size": 640,
  "num_classes": 9,
  "device": "cuda"
}
```

### Data Models (Pydantic)

```python
# Request Models
class DetectionRequest(BaseModel):
    confidence: float = 0.5
    iou_threshold: float = 0.45

# Response Models
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    processing_time_ms: float
    image_size: Tuple[int, int]
```

---

## 🛠️ Technology Stack

### Backend
- **Framework:** FastAPI (Python web framework)
- **Server:** Uvicorn (ASGI server)
- **Async:** AsyncIO (concurrent request handling)
- **Validation:** Pydantic (data validation)

### ML Inference
- **Framework:** PyTorch + Ultralytics
- **Model:** YOLOv12 (from Phase 1)
- **Compute:** GPU support (CUDA/ROCm)

### Containerization
- **Container Runtime:** Docker
- **Orchestration:** Docker Compose (local), Kubernetes (optional)
- **Base Image:** python:3.11-slim

### Testing
- **Framework:** pytest
- **Async Testing:** pytest-asyncio
- **HTTP Testing:** httpx
- **Mocking:** unittest.mock

### Deployment
- **Reverse Proxy:** Nginx
- **Orchestration:** Kubernetes (optional)
- **Monitoring:** Prometheus/Grafana (optional)

---

## 📦 Dependencies

### Core Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
```

### ML/Image Processing
```
torch==2.1.0
ultralytics==8.0.0
opencv-python==4.8.0
pillow==10.0.0
numpy==1.24.0
```

### Utilities
```
python-multipart==0.0.6
aiofiles==23.2.0
```

### Development & Testing
```
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.0
black==23.11.0
pylint==3.0.0
```

**NOTE:** Use the shared `../requirements.txt` for the entire project.
Do NOT create separate requirements.txt in phase2/api/

---

## 🚀 Getting Started with Phase 2

### 1. Model Weights Already Copied ✅
The fine-tuned model is already in `model_weights/best.pt`

### 2. Setup Development Environment
```bash
# Navigate to project root
cd ../

# Create/activate shared virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (shared across project)
pip install -r requirements.txt
```

### 3. Start Phase 2 Development
```bash
cd phase2
# Files are ready for implementation:
# - api/main.py
# - docker/Dockerfile
# - tests/test_api.py
# - deployment/ENV_TEMPLATE
```

### 3. Run API Locally
```bash
cd api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access API Documentation
```
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
```

### 5. Test with Sample Image
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@../phase1/dataset/filtered/test/images/sample.jpg"
```

---

## 🐳 Docker Usage

### Build Container
```bash
docker build -f docker/Dockerfile -t taka-detector:latest .
```

### Run Container Locally
```bash
docker run -p 8000:8000 \
  -v $(pwd)/model_weights:/app/model_weights \
  taka-detector:latest
```

### Using Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up
```

### Access Running Container
```bash
# Check logs
docker logs container_id

# Execute command
docker exec -it container_id bash
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/test_detector.py -v
```

### Run API Tests
```bash
pytest tests/test_api.py -v
```

### Run All Tests
```bash
pytest tests/ -v --cov=api/
```

### Load Testing
```bash
# Using Apache Bench (when needed)
ab -n 1000 -c 10 http://localhost:8000/health
```

---

## 📊 Configuration Management

### Environment Variables (TO CREATE)

```env
# Model Configuration
MODEL_PATH=./model_weights/best.pt
INPUT_SIZE=640
NUM_CLASSES=9

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Inference Configuration
DEFAULT_CONFIDENCE=0.5
DEFAULT_IOU=0.45

# GPU Configuration
USE_GPU=true
GPU_MEMORY_FRACTION=0.8
```

### Configuration File (config.py)
- Load from environment variables
- Support multiple deployment environments (dev, staging, prod)
- Validate configuration on startup

---

## 📚 Phase 2 Implementation Checklist

### API Development
- [ ] Create FastAPI application structure
- [ ] Implement Pydantic data models
- [ ] Create YOLO detector wrapper (detector.py)
- [ ] Implement /health endpoint
- [ ] Implement /detect endpoint
- [ ] Implement /detect/batch endpoint
- [ ] Add error handling & logging
- [ ] Add request validation
- [ ] Implement response formatting
- [ ] Add API documentation

### Docker Setup
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create .dockerignore
- [ ] Create entrypoint.sh
- [ ] Test container build
- [ ] Test container runtime
- [ ] Optimize image size
- [ ] Setup volume mounts for models

### Testing
- [ ] Create pytest configuration
- [ ] Write unit tests (detector.py)
- [ ] Write unit tests (models.py)
- [ ] Write integration tests (API endpoints)
- [ ] Write tests for error handling
- [ ] Create sample test images
- [ ] Setup test fixtures
- [ ] Run full test coverage
- [ ] Document test procedures

### Documentation
- [ ] Complete API_DOCUMENTATION.md
- [ ] Complete DEPLOYMENT.md
- [ ] Complete SETUP.md
- [ ] Complete ARCHITECTURE.md
- [ ] Create TROUBLESHOOTING.md
- [ ] Create ENV_TEMPLATE
- [ ] Document API endpoints with examples
- [ ] Document Docker usage
- [ ] Document deployment procedures

### Deployment
- [ ] Setup CI/CD pipeline
- [ ] Configure GitHub Actions (optional)
- [ ] Create deployment script
- [ ] Setup environment for staging
- [ ] Deploy to production
- [ ] Setup monitoring/logging
- [ ] Create runbooks for ops team

---

## 🔐 Security Considerations

### Image Validation
- [ ] Validate file format (JPEG, PNG only)
- [ ] Check file size limits
- [ ] Validate image dimensions
- [ ] Prevent path traversal attacks

### Rate Limiting
- [ ] Implement rate limiting per IP
- [ ] Add request throttling
- [ ] Implement request queuing

### Input Sanitization
- [ ] Validate parameter types
- [ ] Range check confidence threshold
- [ ] Escape output in responses

### Secrets Management
- [ ] Use environment variables for secrets
- [ ] Don't hardcode credentials
- [ ] Rotate credentials regularly
- [ ] Use secrets management service in production

---

## 📈 Performance Optimization

### Model Inference
- [ ] Load model once (don't reload per request)
- [ ] Use GPU if available
- [ ] Batch inference support
- [ ] Model quantization (optional)

### API Performance
- [ ] Async request handling
- [ ] Connection pooling
- [ ] Request/response compression
- [ ] Caching strategies

### Monitoring Metrics
- [ ] Request latency
- [ ] Inference time
- [ ] GPU utilization
- [ ] Memory usage
- [ ] Error rates

---

## 🔄 Integration with Phase 1

### Model Loading
```python
# Load Phase 1 trained model
from ultralytics import YOLO

model = YOLO('model_weights/best.pt')
# Model automatically in inference mode
# Ready for /detect endpoint
```

### Class Labels
```yaml
# From phase1/dataset/filtered/data.yaml
names:
  0: '500 taka'
  1: 'Fifty taka'
  2: 'Five Taka'
  3: 'One Taka'
  4: 'One Thousand taka'
  5: 'Ten Taka'
  6: 'Twenty'
  7: 'one hundred taka'
  8: 'two taka'
```

### Test Data
- Use test images from Phase 1: `../phase1/dataset/filtered/test/images/`
- Compare Phase 2 predictions with Phase 1 evaluation results
- Validate consistency across phases

---

## 📞 Getting Help

### Before Development
1. Review Phase 1 README: `../phase1/README_PHASE1.md`
2. Understand trained model: `../phase1/training/bd_taka_detector/weights/best.pt`
3. Study Ultralytics YOLO: https://docs.ultralytics.com/

### During Development
1. Check API design specs in this document
2. Review related issues in documentation
3. Test locally before Docker
4. Check logs for error details

### Common Questions
- **Where is the model?** `model_weights/best.pt` (copy from Phase 1)
- **What classes to use?** See class mappings above
- **How to test?** See Testing section
- **How to deploy?** See Phase 2D roadmap

---

## 🎯 Success Criteria

### By Phase 2 Completion

✅ **Functional Requirements:**
- [ ] REST API serves predictions
- [ ] Docker container runs successfully
- [ ] All endpoints documented
- [ ] Error handling implemented
- [ ] Tests pass (>80% coverage)

✅ **Performance Requirements:**
- [ ] Single image inference < 100ms (CPU) / < 50ms (GPU)
- [ ] Batch processing supported
- [ ] Concurrent request handling
- [ ] Memory efficient

✅ **Deployment Requirements:**
- [ ] Production Dockerfile
- [ ] Environment configuration
- [ ] Deployment documentation
- [ ] Rollback procedures
- [ ] Monitoring setup

---

## 📅 Timeline

| Phase | Duration | Deliverables |
|:--|:--|:--|
| 2A | 1-2 weeks | Core API |
| 2B | 1 week | Docker setup |
| 2C | 1-2 weeks | Tests & Docs |
| 2D | 1 week | Deployment |
| **Total** | **4-6 weeks** | **Production API** |

---

## 📧 Contact & Support

For Phase 2 development questions:
1. Review this README
2. Check Phase 1 artifacts at `../phase1/`
3. Consult Ultralytics documentation
4. Check implementation guides in `docs/` (TO CREATE)

---

**Phase 2 Status:** 🔄 IN PREPARATION  
**Start Date:** Ready to begin  
**Repository Structure:** ✅ Ready  
**Phase 1 Integration:** ✅ Ready  
**Previous Phase:** ✅ Phase 1 Complete

