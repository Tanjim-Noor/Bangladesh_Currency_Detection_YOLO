# Bangladeshi Taka Currency Detection - Multi-Phase Project

## 📋 Project Overview

This project implements a **custom object detection system** for recognizing Bangladeshi Taka currency notes using **YOLOv12** model architecture. The project is organized into two distinct phases:

- **Phase 1**: Dataset preparation, model training, and evaluation (COMPLETED)
- **Phase 2**: REST API development, Docker containerization, and deployment (IN PREPARATION)

---

## 📁 Project Structure Overview

```
bangladeshi-taka-detection/
├── README.md                              # This file - Project overview and structure guide
├── requirements.txt                       # Python dependencies
│
├── ========== PHASE 1 - COMPLETED ==========
│
├── 📁 phase1/                             # Phase 1: Training & Evaluation
│   ├── README_PHASE1.md                   # Phase 1 detailed documentation
│   │
│   ├── 📁 dataset/                        # Dataset management
│   │   ├── 📁 original/                   # Original dataset from Roboflow
│   │   │   ├── data.yaml                  # Original dataset config
│   │   │   ├── train/  (1,166 images)
│   │   │   ├── valid/  (168 images)
│   │   │   └── test/   (189 images)
│   │   │
│   │   └── 📁 filtered/                   # Cleaned dataset (11 → 9 classes)
│   │       ├── data.yaml
│   │       ├── train/, valid/, test/
│   │
│   ├── 📁 training/                       # Training artifacts & notebooks
│   │   ├── bangladeshi_taka_detection_yolov12.ipynb
│   │   ├── yolo_model_finetune_showcase.ipynb
│   │   ├── gpu_diagnostics.ipynb
│   │   ├── yolo11n.pt, yolo12n.pt, yolo12m.pt
│   │   ├── 📁 bd_taka_detector/
│   │   │   ├── weights/best.pt            # ⭐ BEST TRAINED MODEL
│   │   │   ├── results.csv
│   │   │   └── confusion_matrix.png
│   │   └── 📁 val_logs/
│   │
│   └── 📁 evaluation/                     # Evaluation results
│       └── inference_results/predictions/
│
│
├── ========== PHASE 2 - IN PREPARATION ==========
│
├── 📁 phase2/                             # Phase 2: Deployment & API
│   ├── README_PHASE2.md                   # Phase 2 setup guide (TO CREATE)
│   │
│   ├── 📁 model_weights/                  # Production model weights
│   │   ├── best.pt                        # ⭐ FINE-TUNED MODEL (from Phase 1)
│   │   ├── yolo12m.pt                     # Pre-trained baseline (reference)
│   │   └── yolo12n.pt                     # Pre-trained baseline (reference)
│   │
│   ├── 📁 api/                            # REST API implementation
│   │   ├── main.py                        # FastAPI app (TO CREATE)
│   │   ├── models.py                      # Data schemas (TO CREATE)
│   │   ├── detector.py                    # Inference wrapper (TO CREATE)
│   │   └── config.py                      # Configuration (TO CREATE)
│   │
│   ├── 📁 docker/                         # Docker containerization
│   │   ├── Dockerfile                     # (TO CREATE)
│   │   ├── docker-compose.yml             # (TO CREATE)
│   │   └── .dockerignore                  # (TO CREATE)
│   │
│   ├── 📁 tests/                          # Testing suite
│   │   ├── test_api.py                    # (TO CREATE)
│   │   ├── test_detector.py               # (TO CREATE)
│   │   └── sample_images/
│   │
│   ├── 📁 deployment/                     # Deployment documentation
│   │   ├── DEPLOYMENT.md                  # (TO CREATE)
│   │   ├── API_DOCUMENTATION.md           # (TO CREATE)
│   │   └── ENV_TEMPLATE                   # (TO CREATE)
│   │
│   └── 📁 docs/                           # Phase 2 documentation
│       ├── SETUP.md                       # (TO CREATE)
│       └── ARCHITECTURE.md                # (TO CREATE)
│
│
├── ========== LEGACY FILES (For Reference) ==========
│
└── Legacy locations (moved to phase1/):
    ├── data/              → phase1/dataset/original
    ├── data_filtered/     → phase1/dataset/filtered
    ├── runs/              → phase1/training
    └── inference_results/ → phase1/evaluation
```

---

## 🎯 Quick Status

### ✅ Phase 1 - COMPLETED

**Status:** All training and evaluation artifacts organized

**Key Deliverables:**
- ✅ Dataset collection (1,523 images, 11 classes)
- ✅ Dataset filtering (reduced to 9 classes)
- ✅ Model training with YOLOv12
- ✅ Model evaluation and testing
- ✅ Training notebooks and documentation

**Location:** `./phase1/`

**Access Trained Model:**
```
Path: ./phase1/training/bd_taka_detector/weights/best.pt
Architecture: YOLOv12
Classes: 9 Bangladeshi currency denominations
```

---

### 🔄 Phase 2 - IN PREPARATION

**Status:** Folder structure created, ready for development

**Objectives:**
- [ ] Develop REST API for model inference
- [ ] Create Docker configuration
- [ ] Build testing suite
- [ ] Implement deployment pipeline

**Location:** `./phase2/`

**Key Dependencies (from Phase 1):**
- Fine-tuned Model: `./phase1/training/bd_taka_detector/weights/best.pt`
- Dataset Config: `./phase1/dataset/filtered/data.yaml`
- Training Details: `./phase1/training/bd_taka_detection_yolov12.ipynb`
- Root Dependencies: `./requirements.txt` (shared across project)

---

## 📊 Dataset Summary

### Currency Classes (9 Categories)
| ID | Class | Value |
|:--|:--|:--|
| 0 | 500 taka | ৳500 |
| 1 | Fifty taka | ৳50 |
| 2 | Five Taka | ৳5 |
| 3 | One Taka | ৳1 |
| 4 | One Thousand taka | ৳1000 |
| 5 | Ten Taka | ৳10 |
| 6 | Twenty | ৳20 |
| 7 | one hundred taka | ৳100 |
| 8 | two taka | ৳2 |

### Data Distribution
| Split | Count | Percentage |
|:--|--:|--:|
| Training | 1,166 | 76.6% |
| Validation | 168 | 11.0% |
| Test | 189 | 12.4% |
| **Total** | **1,523** | **100%** |

### Filtering Details
- **Original Classes:** 11
- **Final Classes:** 9
- **Removed Classes:**
  - "Five Hundred taka" (only 1 sample)
  - "currency" (generic/confusing)

---

## 📂 File Locations Reference

### Phase 1 Critical Files
| File | Location | Purpose |
|:--|:--|:--|
| **Best Model** | `phase1/training/bd_taka_detector/weights/best.pt` | Production inference |
| **Dataset Config** | `phase1/dataset/filtered/data.yaml` | Class labels & paths |
| **Training Notebook** | `phase1/training/bangladeshi_taka_detection_yolov12.ipynb` | Training pipeline |
| **Training Metrics** | `phase1/training/bd_taka_detector/results.csv` | Performance stats |
| **Test Predictions** | `phase1/evaluation/inference_results/predictions/` | Inference outputs |

### Phase 2 Setup Files (To Create)
| Component | Location | Purpose |
|:--|:--|:--|
| **API Server** | `phase2/api/main.py` | FastAPI application |
| **Container** | `phase2/docker/Dockerfile` | Docker configuration |
| **Tests** | `phase2/tests/test_api.py` | API testing |
| **Deploy Guide** | `phase2/deployment/DEPLOYMENT.md` | Deployment instructions |
| **API Docs** | `phase2/deployment/API_DOCUMENTATION.md` | API endpoints |

---

## 🚀 Getting Started

### For Phase 1 Review/Reference
```bash
# Navigate to training folder
cd phase1/training

# View training notebook
jupyter notebook bangladeshi_taka_detection_yolov12.ipynb
```

### For Phase 2 Development
```bash
# Navigate to Phase 2
cd phase2

# Setup API development
cd api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install API dependencies (TO CREATE)
pip install -r requirements.txt
```

---

## 🔗 Phase Integration Map

```
PHASE 1 (COMPLETED)
├── Outputs:
│   ├── Trained model: best.pt
│   ├── Dataset config: data.yaml
│   ├── Class mappings: 0-8
│   └── Training metrics: results.csv
│
└──→ PHASE 2 (IN PREPARATION)
     ├── Loads: best.pt for inference
     ├── Uses: data.yaml for class labels
     ├── Builds: REST API wrapper
     ├── Tests: With Phase 1 test set
     └── Deploys: Via Docker container
```

---

## 📚 Documentation

| Document | Location | Status |
|:--|:--|:--|
| **Project Overview** | README.md (this file) | ✅ Done |
| **Phase 1 Details** | phase1/README_PHASE1.md | 🔄 To Create |
| **Phase 2 Setup** | phase2/README_PHASE2.md | 🔄 To Create |
| **API Documentation** | phase2/deployment/API_DOCUMENTATION.md | 🔄 To Create |
| **Deployment Guide** | phase2/deployment/DEPLOYMENT.md | 🔄 To Create |
| **System Architecture** | phase2/docs/ARCHITECTURE.md | 🔄 To Create |

---

## 🛠️ Tech Stack

### Phase 1 (Completed)
- **Framework:** PyTorch, Ultralytics
- **Model:** YOLOv12
- **Data:** NumPy, Pillow, OpenCV
- **Development:** Jupyter Notebooks
- **Compute:** GPU (CUDA)

### Phase 2 (To Implement)
- **API:** FastAPI or Flask
- **Server:** Uvicorn/Gunicorn
- **Container:** Docker
- **Testing:** pytest, httpx
- **CI/CD:** GitHub Actions (optional)

---

## 📦 Dependencies

### Phase 1
```
ultralytics
supervision
matplotlib
numpy
Pillow
pyyaml
ipykernel
```

See `requirements.txt` for complete Phase 1 dependencies.

### Phase 2 (Additional)
```
fastapi
uvicorn
pydantic
python-multipart
opencv-python
pillow
pytest
httpx
docker
```

---

## ✅ Completion Checklist

### Phase 1
- ✅ Dataset collected from Roboflow
- ✅ Dataset annotated in YOLO format
- ✅ Classes analyzed and filtered
- ✅ Model trained with YOLOv12
- ✅ Evaluation performed
- ✅ Results documented
- ✅ Artifacts organized in phase1/

### Phase 2 (Preparation)
- ✅ Folder structure created
- ✅ Model weights copied
- ✅ Placeholders for all components
- ✅ Documentation structure ready
- ⏳ Implementation (to start)

---

## 📧 Project Metadata

**Assignment:** Model Fine-tuning & Intro to Transformers  
**Course/Institution:** Ostad  
**Dataset Source:** [Roboflow Universe](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection/dataset/3)  
**Dataset License:** CC BY 4.0  
**Model:** YOLOv12  
**Last Updated:** January 2026  

---

## 🔍 Key Findings from Phase 1

### Dataset Quality
- Successfully filtered out problematic classes
- Maintained full dataset size (no images removed)
- Clean annotations in YOLO format
- Balanced class distribution

### Model Training
- YOLOv12 selected for its attention-centric architecture
- Trained on filtered 9-class dataset
- Best model saved with checkpoint mechanism
- Full training history recorded

### Evaluation Results
- Complete test set evaluation
- Inference predictions generated
- Confusion matrix analysis available
- Performance metrics documented

---

## 📞 Next Steps

### For Phase 2 Development:
1. Review Phase 1 artifacts in `phase1/` folder
2. Load best model: `phase1/training/bd_taka_detector/weights/best.pt`
3. Reference dataset config: `phase1/dataset/filtered/data.yaml`
4. Implement REST API in `phase2/api/`
5. Build Docker container in `phase2/docker/`
6. Add tests in `phase2/tests/`
7. Complete deployment documentation in `phase2/deployment/`

---

**Project Status:** Phase 1 Complete ✅ | Phase 2 Ready to Start 🔄
