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

**Original Dataset:** ~1,770 images with 11 classes

**Data Filtering:** Excludes underrepresented classes for better model performance
- **Excluded Classes:** 
  - Five Hundred taka (only 1 sample - insufficient for training)
  - currency (71 samples - generic class)
- **Final Classes:** 9 denominations

### Classes (9 categories - After Filtering)
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

### Dataset Split (After Filtering)
| Split | Images | Percentage |
|-------|--------|------------|
| Training | ~1,434 | 80% |
| Validation | ~178 | 10% |
| Test | ~179 | 10% |
| **Total** | **~1,791** | 100% |

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

## 👤 Author
**Student Assignment - Model Finetuning & Intro to Transformers**

---

*Last Updated: December 2024*