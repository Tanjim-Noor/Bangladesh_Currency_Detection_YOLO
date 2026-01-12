# Phase 1: Bangladeshi Taka Detection - Training & Evaluation

## 📋 Phase 1 Overview

Phase 1 encompasses all dataset preparation, model training, and evaluation activities for the Bangladeshi Taka Currency Detection project. All Phase 1 artifacts are organized in this folder for easy reference and as foundation for Phase 2 development.

**Status:** ✅ COMPLETED

---

## 📁 Phase 1 Folder Structure

```
phase1/
├── README_PHASE1.md              # This file
│
├── dataset/                      # Dataset management
│   ├── original/                 # Original Roboflow dataset
│   │   ├── data.yaml             # Original dataset config (11 classes)
│   │   ├── train/
│   │   │   ├── images/           # 1,166 training images
│   │   │   └── labels/           # YOLO format annotations
│   │   ├── valid/
│   │   │   ├── images/           # 168 validation images
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/           # 189 test images
│   │       └── labels/
│   │
│   └── filtered/                 # Cleaned dataset (9 classes)
│       ├── data.yaml             # Filtered dataset config
│       ├── train/
│       ├── valid/
│       └── test/
│
├── training/                     # Training artifacts
│   ├── bangladeshi_taka_detection_yolov12.ipynb  # Main training
│   ├── yolo_model_finetune_showcase.ipynb        # Reference
│   ├── gpu_diagnostics.ipynb     # GPU verification
│   │
│   ├── yolo11n.pt                # Pretrained weights
│   ├── yolo12n.pt
│   ├── yolo12m.pt
│   │
│   ├── bd_taka_detector/         # Training output
│   │   ├── weights/
│   │   │   ├── best.pt           # ⭐ BEST MODEL
│   │   │   └── last.pt           # Last checkpoint
│   │   ├── results.csv           # Training metrics
│   │   ├── results.png           # Training curves
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── args.yaml             # Training configuration
│   │
│   └── val_logs/                 # Validation run results
│
└── evaluation/                   # Evaluation results
    └── inference_results/
        ├── predictions/          # Test set predictions
        │   ├── image_with_boxes/ # Annotated images
        │   └── runs/             # Detection outputs
        └── metrics.json          # Evaluation metrics
```

---

## 📊 Dataset Information

### Dataset Source
- **Platform:** Roboflow Universe
- **Dataset Link:** https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection/dataset/3
- **License:** CC BY 4.0
- **Format:** YOLO (object detection format)

### Original Dataset Characteristics
- **Total Images:** 1,523
- **Original Classes:** 11
- **Format:** YOLO txt format annotations

#### Original Class List
1. 500 taka - ৳500 denomination
2. Fifty taka - ৳50 denomination
3. Five Hundred taka - ৳500 (REMOVED)
4. Five Taka - ৳5 denomination
5. One Taka - ৳1 denomination
6. One Thousand taka - ৳1,000 denomination
7. Ten Taka - ৳10 denomination
8. Twenty - ৳20 denomination
9. currency - Generic label (REMOVED)
10. one hundred taka - ৳100 denomination
11. two taka - ৳2 denomination

### Dataset Filtering Process

#### Why Filtering Was Necessary
The original dataset contained two problematic classes:

**1. "Five Hundred taka" Class**
- **Issue:** Only 1 sample in the entire dataset
- **Problem:** Insufficient training data for neural network to learn
- **Solution:** Excluded from training
- **Impact:** No images removed (they contained other denominations)

**2. "currency" Class**
- **Issue:** Generic, overlapping with specific denominations
- **Problem:** Ambiguous annotations causing model confusion
- **Solution:** Removed all annotations for this class
- **Impact:** No images removed (retained for other denominations)

#### Filtering Results
- **Annotations Removed:** 135 total
  - Train: 72 annotations
  - Valid: 37 annotations
  - Test: 26 annotations
- **Images Removed:** 0 (all images retained)
- **Final Classes:** 9 (11 → 9)

### Filtered Dataset Characteristics

#### Final Currency Classes (9 Classes)
| ID | Class | Denomination | Training Count |
|:--|:--|:--|--:|
| 0 | 500 taka | ৳500 | 129 |
| 1 | Fifty taka | ৳50 | 335 |
| 2 | Five Taka | ৳5 | 478 |
| 3 | One Taka | ৳1 | 236 |
| 4 | One Thousand taka | ৳1,000 | 194 |
| 5 | Ten Taka | ৳10 | 361 |
| 6 | Twenty | ৳20 | 408 |
| 7 | one hundred taka | ৳100 | 435 |
| 8 | two taka | ৳2 | 355 |

#### Data Split Distribution
| Split | Images | Percentage |
|:--|--:|--:|
| Training | 1,166 | 76.6% |
| Validation | 168 | 11.0% |
| Test | 189 | 12.4% |
| **Total** | **1,523** | **100%** |

---

## 🤖 Model Information

### YOLOv12 Overview
**YOLOv12** is an attention-centric, real-time object detection architecture.

#### Model Details
- **Architecture:** Attention-Centric YOLO
- **Release:** February 2025
- **Paper:** [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)
- **Framework:** PyTorch + Ultralytics

#### Key Features
- ✅ Improved accuracy with attention mechanisms
- ✅ Faster inference than previous versions
- ✅ Better convergence during training
- ✅ Lower computational overhead
- ✅ Multiple scale variants (nano, small, medium, large)

#### Variants Used
- **yolo12n.pt** - Nano (smallest, fastest)
- **yolo12m.pt** - Medium (balanced)

### Training Configuration

| Parameter | Value |
|:--|:--|
| **Model** | YOLOv12 |
| **Pretrained Weights** | yolo12m.pt (ImageNet weights) |
| **Epochs** | 50 |
| **Batch Size** | 8 |
| **Image Size** | 640×640 pixels |
| **Optimizer** | AdamW (automatic) |
| **Initial Learning Rate** | 0.01 |
| **Final Learning Rate** | 0.0001 |
| **Warmup Epochs** | 3 |
| **Device** | GPU (CUDA) |
| **Mixed Precision** | Enabled (fp16) |
| **Dataset** | Filtered (9 classes) |
| **Augmentation** | Default (flip, mosaic, etc.) |
| **Early Stopping** | Enabled (patience=20) |

---

## 🎯 Training Process

### Step 1: Dataset Preparation
1. Downloaded original dataset from Roboflow (1,523 images)
2. Analyzed class distribution
3. Identified and filtered problematic classes
4. Verified YOLO format annotations

### Step 2: Environment Setup
1. GPU diagnostics (see `gpu_diagnostics.ipynb`)
2. Installed CUDA-compatible PyTorch
3. Verified GPU availability
4. Set up Ultralytics YOLO

### Step 3: Model Training
1. Loaded pretrained YOLOv12m weights
2. Configured training parameters
3. Trained for 50 epochs
4. Monitored validation metrics
5. Saved best model checkpoint

### Step 4: Evaluation
1. Evaluated best model on test set
2. Generated predictions for all test images
3. Computed evaluation metrics
4. Created confusion matrix
5. Generated annotated predictions

---

## 📈 Training Metrics & Results

### Best Model Performance
**File:** `training/bd_taka_detector/weights/best.pt`

#### Key Metrics
- **Training Loss:** Progressive decrease across epochs
- **Validation Loss:** Stable convergence
- **Box Loss:** Bounding box prediction accuracy
- **Classification Loss:** Class prediction accuracy
- **Confidence Loss:** Object detection confidence

See `training/bd_taka_detector/results.csv` for detailed epoch-by-epoch metrics.

### Evaluation Results
- **Test Set Size:** 189 images
- **Test Predictions:** All test images processed
- **Output Location:** `evaluation/inference_results/predictions/`

#### Predictions Include
- Annotated images with bounding boxes
- Detection confidence scores
- Class predictions per detected object
- Visualization of model performance

### Confusion Matrix
Located at: `training/bd_taka_detector/confusion_matrix.png`
- Shows per-class precision/recall
- Helps identify misclassification patterns
- Useful for improving future training

---

## 📁 Key Files Reference

### Dataset Files
| File | Location | Purpose |
|:--|:--|:--|
| Original Config | `dataset/original/data.yaml` | Original dataset paths (11 classes) |
| Filtered Config | `dataset/filtered/data.yaml` | Training dataset config (9 classes) |
| Training Images | `dataset/filtered/train/images/` | 1,166 training images |
| Validation Images | `dataset/filtered/valid/images/` | 168 validation images |
| Test Images | `dataset/filtered/test/images/` | 189 test images |

### Training Files
| File | Location | Purpose |
|:--|:--|:--|
| Main Notebook | `training/bangladeshi_taka_detection_yolov12.ipynb` | Training pipeline |
| GPU Check | `training/gpu_diagnostics.ipynb` | GPU setup verification |
| Pretrained Weights | `training/yolo12m.pt` | Starting weights |
| **Best Model** | `training/bd_taka_detector/weights/best.pt` | **Final trained model** |
| Metrics CSV | `training/bd_taka_detector/results.csv` | Training statistics |
| Config YAML | `training/bd_taka_detector/args.yaml` | Training parameters |

### Evaluation Files
| File | Location | Purpose |
|:--|:--|:--|
| Test Predictions | `evaluation/inference_results/predictions/` | Annotated test outputs |
| Confusion Matrix | `training/bd_taka_detector/confusion_matrix.png` | Classification analysis |
| Training Curves | `training/bd_taka_detector/results.png` | Loss & accuracy plots |

---

## 🚀 How to Use Phase 1 Artifacts

### Reviewing Training Results
```bash
# 1. Open main training notebook
cd phase1/training
jupyter notebook bangladeshi_taka_detection_yolov12.ipynb

# 2. Review training metrics
cat bd_taka_detector/results.csv

# 3. View confusion matrix
open bd_taka_detector/confusion_matrix.png
```

### Loading the Trained Model
```python
from ultralytics import YOLO

# Load best trained model
model = YOLO('phase1/training/bd_taka_detector/weights/best.pt')

# Perform inference
results = model.predict('image.jpg')
```

### Accessing Dataset Configuration
```python
import yaml

# Load dataset config
with open('phase1/dataset/filtered/data.yaml', 'r') as f:
    dataset_config = yaml.safe_load(f)

# Get class names
classes = dataset_config['names']  # {0: '500 taka', 1: 'Fifty taka', ...}
```

### Running Predictions on Test Set
```python
from ultralytics import YOLO
import os

model = YOLO('phase1/training/bd_taka_detector/weights/best.pt')

# Get test images
test_images = os.listdir('phase1/dataset/filtered/test/images/')

# Run predictions
for img in test_images:
    results = model.predict(f'phase1/dataset/filtered/test/images/{img}')
```

---

## 📊 Class Label Reference

### Currency Denomination Mapping
For Phase 2 API development, use this mapping:

```yaml
class_id_to_name:
  0: "500 taka"
  1: "Fifty taka"
  2: "Five Taka"
  3: "One Taka"
  4: "One Thousand taka"
  5: "Ten Taka"
  6: "Twenty"
  7: "one hundred taka"
  8: "two taka"

name_to_class_id:
  "500 taka": 0
  "Fifty taka": 1
  "Five Taka": 2
  "One Taka": 3
  "One Thousand taka": 4
  "Ten Taka": 5
  "Twenty": 6
  "one hundred taka": 7
  "two taka": 8
```

---

## 🔄 Transition to Phase 2

### Assets Available for Phase 2

**Model:**
- Trained weights: `training/bd_taka_detector/weights/best.pt`
- Compatible with YOLOv12 inference

**Configuration:**
- Dataset config: `dataset/filtered/data.yaml`
- Class mappings: 9 classes (0-8)
- Training parameters: `training/bd_taka_detector/args.yaml`

**Reference Data:**
- Test set: `dataset/filtered/test/`
- Test predictions: `evaluation/inference_results/`
- Training curves: `training/bd_taka_detector/results.png`

**Notebooks:**
- Training pipeline: `training/bangladeshi_taka_detection_yolov12.ipynb`
- Model fine-tuning reference: `training/yolo_model_finetune_showcase.ipynb`

### Phase 2 Development Starting Points

1. **Copy Model to Phase 2:**
   ```bash
   cp training/bd_taka_detector/weights/best.pt ../phase2/model_weights/best.pt
   ```

2. **Reference Dataset Config:**
   - Use `dataset/filtered/data.yaml` for class labels
   - Store class mappings in Phase 2 API config

3. **Test With Phase 1 Test Set:**
   - Use `dataset/filtered/test/` for API testing
   - Verify predictions match Phase 1 results

4. **Document Model Details:**
   - Note: YOLOv12 architecture
   - Input size: 640×640
   - 9 output classes
   - Confidence threshold suggestions from Phase 1

---

## 💡 Key Insights

### Dataset Quality
- Filtering process improved data quality without image loss
- Removed 2 problematic classes (only annotations removed)
- Final dataset well-balanced across 9 classes

### Model Performance
- YOLOv12 showed stable convergence
- Pretrained ImageNet weights provided good initialization
- 50 epochs sufficient for convergence
- Mixed precision training reduced memory usage

### Best Practices Applied
- Separate original/filtered dataset copies
- Checkpoint mechanism for best model
- Comprehensive logging of training metrics
- Confusion matrix for error analysis

---

## 📚 Notebook Summaries

### Main Training Notebook: `bangladeshi_taka_detection_yolov12.ipynb`
**Purpose:** Complete training pipeline for YOLOv12 on Bangladeshi Taka dataset

**Sections:**
1. Environment Setup & GPU Check
2. Dataset Loading & Exploration
3. Data Augmentation Preview
4. Model Training (50 epochs)
5. Evaluation on Test Set
6. Inference & Visualization
7. Model Export & Metrics

### GPU Diagnostics: `gpu_diagnostics.ipynb`
**Purpose:** Verify GPU setup and CUDA compatibility

**Checks:**
- CUDA availability
- PyTorch GPU access
- GPU memory
- CUDA version compatibility

### Reference Notebook: `yolo_model_finetune_showcase.ipynb`
**Purpose:** Example of fine-tuning YOLOv12

**Content:**
- Fine-tuning techniques
- Custom dataset handling
- Model evaluation methods

---

## ✅ Phase 1 Completion Checklist

- ✅ Dataset collected from Roboflow (1,523 images)
- ✅ Dataset analyzed for quality (11 → 9 classes)
- ✅ Problem classes identified and filtered
- ✅ Dataset split: train/valid/test
- ✅ YOLOv12 model selected and downloaded
- ✅ Training environment configured
- ✅ 50-epoch training completed
- ✅ Best model checkpointed
- ✅ Test set evaluation completed
- ✅ Predictions generated for all test images
- ✅ Confusion matrix analysis performed
- ✅ Training artifacts organized
- ✅ Phase 1 documentation complete

---

## 📞 Support & References

### Training Issues?
- Check `gpu_diagnostics.ipynb` for GPU setup
- Review `bangladeshi_taka_detection_yolov12.ipynb` for training details
- See `bd_taka_detector/args.yaml` for exact configuration

### Dataset Questions?
- Original config: `dataset/original/data.yaml` (11 classes)
- Filtered config: `dataset/filtered/data.yaml` (9 classes)
- Dataset documentation: `dataset/original/README.dataset.txt`

### Model Details?
- Training metrics: `training/bd_taka_detector/results.csv`
- Confusion matrix: `training/bd_taka_detector/confusion_matrix.png`
- Training curves: `training/bd_taka_detector/results.png`

---

**Phase 1 Status:** ✅ COMPLETED  
**Last Updated:** January 2026  
**Next Phase:** Phase 2 - REST API Development
