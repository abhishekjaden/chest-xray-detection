# Chest X-Ray Abnormality Detection

An end-to-end deep learning system that detects 14 thoracic abnormalities in chest X-rays, built with a Faster R-CNN object detector and deployed as a live, interactive web demo.

> ⚠️ **Educational / research demonstration only. Not a diagnostic device. Not validated for clinical use and must not be used for medical decisions.**

## 🔗 Live Demo

**Try it here:** [huggingface.co/spaces/rocky17435/xray-detection](https://huggingface.co/spaces/rocky17435/xray-detection)

Upload a chest X-ray (PNG/JPG) and the model returns detected findings with bounding boxes and confidence scores.

## Overview

This project covers the full machine-learning lifecycle — data engineering, model training, evaluation, API development, a frontend, and live deployment — for detecting thoracic abnormalities in chest radiographs from the [VinBigData Chest X-ray dataset](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection).

The 14 detected findings: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, and Pulmonary fibrosis.

## Architecture

| Layer | Technology |
|-------|-----------|
| Model | Faster R-CNN (ResNet-50 FPN backbone), PyTorch |
| Label fusion | Weighted Boxes Fusion (multi-radiologist consensus) |
| Inference | Test-Time Augmentation (horizontal flip + WBF) |
| API | FastAPI (predict, DICOM support, annotated-image endpoint) |
| Frontend | React + Vite (interactive canvas, threshold slider, class filters) |
| Deployment | Gradio app on Hugging Face Spaces |

## Results

Evaluated on a held-out test set (660 images):

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.29 |
| mAP@0.5:0.95 | 0.13 |

This is comparable to published competition baselines for this dataset.

### Per-class performance (mAP@0.5:0.95)

Performance tracks training-data volume closely — a central finding of this project:

| Strong (well-represented) | Score | Weak (data-scarce) | Score |
|---------------------------|-------|--------------------|-------|
| Cardiomegaly | 0.54 | Other lesion | 0.01 |
| Aortic enlargement | 0.51 | Calcification | 0.03 |
| Pleural effusion | 0.10 | Pleural thickening | 0.03 |

The model performs reliably on common findings (which had 700–970 training examples) and poorly on rare classes (some with only 15–30 examples). This **class imbalance**, not the architecture, is the primary performance limiter.

## Key engineering decisions

- **Weighted Boxes Fusion** to merge overlapping multi-radiologist annotations into consensus labels (reduced 36,096 raw boxes to 22,719).
- **Class-balanced sampling** to give rare findings more training exposure.
- **Test-Time Augmentation** improved mAP@0.5:0.95 from 0.126 to 0.132 with no retraining.
- **Honest scoping:** evaluated and reported the model's real limitations rather than overstating performance.

## Limitations

- Trained on a 3,000-image subset; rare classes are under-represented and unreliable.
- Not clinically validated; no regulatory clearance. Educational demonstration only.
- Performance on real-world X-rays outside the dataset distribution may differ.

## Repository structure

- `model.py`, `main.py`, `requirements.txt` — FastAPI inference service
- `src/`, `package.json`, `vite.config.js`, `index.html` — React frontend
- Live model weights are hosted on the Hugging Face Space (not in this repo due to size)

## Tech stack

Python · PyTorch · torchvision · FastAPI · React · Vite · Gradio · Hugging Face Spaces
