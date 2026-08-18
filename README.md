# Chest X-Ray Abnormality Detection

[![Tests](https://github.com/abhishekjaden/chest-xray-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/abhishekjaden/chest-xray-detection/actions/workflows/tests.yml)

An end-to-end deep learning system that detects 14 thoracic abnormalities in chest X-rays — with a rigorous evaluation of *why* it performs as it does.

> ⚠️ **Educational / research demonstration only. Not a diagnostic device. Not validated for clinical use and must not be used for medical decisions.**

## 🔗 Live Demo

**[huggingface.co/spaces/rocky17435/xray-detection](https://huggingface.co/spaces/rocky17435/xray-detection)**

Upload a chest X-ray and the model returns detected findings with bounding boxes and **calibrated** confidence scores.

## 📄 [Read the Technical Report →](TECHNICAL_REPORT.md)

The full study: methodology, per-class analysis, calibration, explainability, and a controlled architecture comparison.

---

## Overview

Multi-class object detection over 14 thoracic findings, using the VinBigData Chest X-ray dataset. The project covers the full lifecycle — data engineering, training, evaluation, API, frontend, and live deployment — but the substantive contribution is the evaluation: An initial hypothesis that data scarcity drove the per-class failures was tested against the full dataset annotations and **rejected** (ρ = +0.055, p = 0.85). Neither data volume nor architecture explains which classes fail.

## Results

Both models evaluated on the same held-out test split (660 images), identical inputs and metric:

| | Faster R-CNN | YOLOv8s |
|---|---|---|
| Parameters | 41.3M | 11.1M |
| mAP@0.5 | 0.290 | **0.348** |
| mAP@0.5:0.95 | 0.132 | **0.180** |

![Architecture comparison](analysis/architecture_comparison.png)

YOLOv8s wins on all 14 classes with ~27% of the parameters — yet the same twelve classes remain unusable in both (Calcification 0.036, Other lesion 0.029). A better architecture lifts performance across the board without changing which findings the model can actually detect.

## Key findings

**Performance is not predicted by training-set size.** Spearman ρ = +0.055 (p = 0.85) across the 14 classes. Pleural thickening has 2,010 training images and scores 0.063 mAP@0.5:0.95; Cardiomegaly has 2,316 and scores 0.655. An initial hypothesis that data scarcity drove the failures was tested against the full dataset annotations and rejected. What does explain the gap is not yet established — positional spread of annotations is the strongest candidate (ρ = −0.433) but does not reach significance at n = 14.

**Both models were badly miscalibrated — in opposite directions.** Faster R-CNN was over-confident: at 75% claimed confidence it was correct 40% of the time. Temperature scaling failed (0.7% improvement) because the miscalibration is non-uniform across the confidence range; isotonic regression reduced calibration error by **91%**. YOLOv8s showed the reverse problem — systematic *under*-confidence, with a raw 0.35 score correct 62% of the time — and isotonic calibration reduced its ECE by **93.7%**. Calibrated output is capped at 0.95 to avoid claiming a certainty the data doesn't support, and is live in the deployed demo.

![Reliability diagram](analysis/calibration_reliability_diagram.png)

**Failure analysis and attention mapping** independently confirm the same picture: the model localises the two well-performing findings accurately and produces false positives on the rest.

## Architecture

| Layer | Technology |
|-------|-----------|
| Models | Faster R-CNN (ResNet-50 FPN) and YOLOv8s, PyTorch |
| Label fusion | Weighted Boxes Fusion (multi-radiologist consensus) |
| Calibration | Isotonic regression, fitted on held-out validation |
| Deployed model | YOLOv8s, single-pass inference (~7 ms/image) |
| Inference (Faster R-CNN) | Test-Time Augmentation (hflip + WBF) |
| API | FastAPI — JSON detections, annotated images, DICOM support |
| Frontend | React + Vite — canvas rendering, threshold slider, class filters |
| Deployment | Gradio on Hugging Face Spaces |

## Engineering practices

- 7 endpoint tests covering happy paths and error cases (`test_api.py`)
- Input validation: size limits, extension checks, graceful error handling
- Calibration validated on held-out data before deployment
- Version-independent artifact storage (JSON curve rather than pickled model)
- Honest scoping and documented limitations throughout
- ONNX export verified bit-exact against the reference implementation, then rejected on measured evidence rather than deployed by default

## Limitations

- Twelve of fourteen classes are effectively undetectable; the cause is not established, and is not training-set size
- Not clinically validated; no regulatory clearance — educational demonstration only
- Calibration is fitted to this dataset's distribution and would need refitting elsewhere
- Out-of-distribution performance untested

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and ethical considerations.

## Repository structure

```
├── TECHNICAL_REPORT.md   full study
├── MODEL_CARD.md         intended use, limitations, ethics
├── analysis/             figures and result data
├── model.py, main.py     FastAPI inference service
├── test_api.py           endpoint tests
└── src/                  React frontend
```

Model weights are hosted on the Hugging Face Space (not in this repo, due to size).

## Tech stack

Python · PyTorch · torchvision · Ultralytics · FastAPI · React · Vite · Gradio · Hugging Face Spaces
