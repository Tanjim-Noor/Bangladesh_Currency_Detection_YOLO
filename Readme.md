# Bangladeshi Taka Currency Detection using YOLOv12

## 📋 Project Overview
This project implements a **custom object detection system** for recognizing Bangladeshi Taka currency notes using the state-of-the-art **YOLOv12** model architecture.

### Assignment: Model Fine-tuning & Intro to Transformers

---

## 📁 Dataset Information

### Dataset Source
- **Platform**: Roboflow Universe
- **Dataset Link**: [https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection/dataset/3](https://universe.roboflow.com/tanvirtain/bangladeshi-currency-detection/dataset/3)
- **License**: CC BY 4.0

### Dataset Information

#### Original Dataset (Before Filtering)
- **Total Images**: 1,523
- **Number of Classes**: 11

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 1,166 | 76.6% |
| Validation | 168 | 11.0% |
| Test | 189 | 12.4% |

#### Original Class Distribution (Training Set)
| Class Name | Count | Issue |
|------------|-------|-------|
| 500 taka | 129 | ✅ Good representation |
| Fifty taka | 335 | ✅ Good representation |
| **Five Hundred taka** | **1** | ❌ **EXCLUDED** - Only 1 sample, insufficient for training |
| Five Taka | 478 | ✅ Good representation |
| One Taka | 236 | ✅ Good representation |
| One Thousand taka | 194 | ✅ Good representation |
| Ten Taka | 361 | ✅ Good representation |
| Twenty | 408 | ✅ Good representation |
| **currency** | **71** | ❌ **EXCLUDED** - Generic class causing confusion |
| one hundred taka | 435 | ✅ Good representation |
| two taka | 355 | ✅ Good representation |

#### Why Filtering Was Done

The filtering process was essential for improving model performance:

1. **Insufficient Training Data**: The "Five Hundred taka" class had only **1 sample**, which is far too small to train a robust detector. Neural networks require hundreds of examples per class to learn meaningful patterns.

2. **Generic/Confusing Class**: The "currency" class was too generic and overlapped with other specific denominations, causing the model to learn ambiguous patterns and reducing overall accuracy.

3. **No Image Removal**: Importantly, **no complete images were removed**. Most images contained multiple currency denominations. The filtering only removed annotations for the problematic classes:
   - Train: 72 annotations removed (images retained)
   - Valid: 37 annotations removed (images retained)
   - Test: 26 annotations removed (images retained)

This approach maintains the dataset's size while ensuring the model trains only on reliable, well-represented classes.

#### Filtered Dataset (After Filtering)
- **Total Images**: 1,523 (No images removed, only annotations cleaned)
- **Number of Classes**: 9 (2 classes excluded)
- **Annotations Removed**: 135 total (72 train, 37 valid, 26 test)

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 1,166 | 76.6% |
| Validation | 168 | 11.0% |
| Test | 189 | 12.4% |

### Classes (9 Categories - After Filtering)
| ID | Class Name |
|----|------------|
| 0 | 500 taka |
| 1 | Fifty taka |
| 2 | Five Taka |
| 3 | One Taka |
| 4 | One Thousand taka |
| 5 | Ten Taka |
| 6 | Twenty |
| 7 | one hundred taka |
| 8 | two taka |

---

## 🏗️ Project Structure
```
Assignment/
├── 📄 Readme.md                              # This file
├── 📓 bangladeshi_taka_detection_yolov12.ipynb  # Main training notebook
├── 📓 yolo_model_finetune_showcase.ipynb     # Reference notebook
│
├── 📁 data/                                   # Dataset folder
│   ├── data.yaml                             # Dataset configuration
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── train/
│   │   ├── images/                           # Training images
│   │   └── labels/                           # YOLO format annotations
│   ├── valid/
│   │   ├── images/                           # Validation images
│   │   └── labels/
│   └── test/
│       ├── images/                           # Test images
│       └── labels/
│
└── 📁 runs/                                   # Training outputs
    └── detect/
        └── bd_taka_detector/
            ├── weights/
            │   ├── best.pt                   # Best model weights
            │   └── last.pt                   # Last checkpoint
            ├── results.png                   # Training curves
            ├── confusion_matrix.png
            └── ...
```

---

## 🚀 Model Training

### Model Selection
- **Architecture**: YOLOv12 (Attention-Centric Real-Time Object Detector)
- **Variant**: YOLOv12n (Nano) / YOLOv12s (Small)
- **Paper**: [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/abs/2502.12524)
- **Release Date**: February 2025

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 8 |
| Image Size | 640×640 |
| Optimizer | AdamW (auto) |
| Learning Rate | 0.01 (initial) |
| Device | GPU (CUDA) |
| Classes | 9 (filtered dataset) |
| Dataset | Filtered (excluded underrepresented classes) |

### YOLOv12 Features
- ✅ Attention-centric architecture for better feature extraction
- ✅ Lower latency compared to previous versions
- ✅ Higher mAP on COCO benchmark
- ✅ Efficient training with optimized convergence

---

## 📊 Evaluation Metrics

The model is evaluated using the following metrics:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision at IoU thresholds from 0.5 to 0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

---

## 💻 Requirements

```bash
# Install dependencies
pip install ultralytics
pip install supervision
pip install matplotlib pillow numpy pyyaml
```

---

## 🎯 How to Run

1. **Open the notebook**:
   ```bash
   jupyter notebook bangladeshi_taka_detection_yolov12.ipynb
   ```

2. **Run all cells** to:
   - Explore the dataset
   - Train the YOLOv12 model
   - Evaluate on test set
   - Generate inference results

3. **Check outputs** in:
   - `runs/detect/bd_taka_detector/` - Training results
   - `inference_results/` - Test predictions

---

## 📝 Assignment Tasks Completed

| Task | Status |
|------|--------|
| 1. Dataset Collection | ✅ Complete |
| 2. Data Annotation & Preparation | ✅ Complete |
| 3. Model Training | ✅ Complete |
| 4. Model Evaluation | ✅ Complete |
| 5. Results & Submission | ✅ Complete |

---

## 📚 References

1. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
2. [YOLOv12 Paper](https://arxiv.org/abs/2502.12524)
3. [Roboflow Universe](https://universe.roboflow.com/)
4. [YOLOv12 Training Guide](https://blog.roboflow.com/train-yolov12-model/)

---