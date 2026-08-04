# Model Card: Chest X-Ray Abnormality Detection

## Model Overview

Object detectors that localize 14 thoracic abnormalities in chest radiographs, trained on the VinBigData Chest X-ray dataset and deployed as an educational demonstration.

- **Task:** Multi-class object detection
- **Models:** YOLOv8s (11.1M params) — **deployed**; Faster R-CNN (ResNet-50 FPN, 41.3M params) — evaluated for comparison
- **Framework:** PyTorch / torchvision / Ultralytics
- **Input:** Chest X-ray (PNG, JPG, or DICOM), resized to 512×512
- **Output:** Bounding boxes with class labels and isotonic-calibrated confidence scores
- **Live demo:** https://huggingface.co/spaces/rocky17435/xray-detection
- **Full study:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

## Intended Use

**Intended:** Educational and research demonstration of an end-to-end medical-imaging ML system, and of evaluation methodology — per-class analysis, confidence calibration, failure analysis, and controlled architecture comparison.

**NOT intended:** Clinical diagnosis, screening, triage, or any medical decision-making. This is **not** a medical device, has **no** regulatory clearance, and must **never** be used to inform patient care.

## Training Data

- **Source:** VinBigData Chest X-ray Abnormalities Detection dataset
- **Subset:** 3,075 train / 659 validation / 660 test images (stratified)
- **Labels:** Multi-radiologist annotations merged via Weighted Boxes Fusion (36,096 raw → 22,719 consensus boxes)
- **Imbalance:** Severe, roughly 65:1. Aortic enlargement appears in 974 training images; Atelectasis in 15.

## Performance

Both models on the same held-out test split:

| | Faster R-CNN | YOLOv8s |
|---|---|---|
| mAP@0.5 | 0.290 | 0.348 |
| mAP@0.5:0.95 | 0.132 | 0.180 |

### Per-class (mAP@0.5:0.95, YOLOv8s)

| Usable | Score | Unusable | Score |
|---|---|---|---|
| Cardiomegaly | 0.655 | Other lesion | 0.029 |
| Aortic enlargement | 0.590 | Calcification | 0.036 |
| Pleural effusion | 0.170 | Pleural thickening | 0.063 |

Two of fourteen classes perform respectably. The remainder are unreliable.

## Calibration

Raw confidence scores were substantially over-confident: predictions claiming 75% confidence were correct 40% of the time (ECE = 0.155 for scores ≥ 0.25).

Temperature scaling was ineffective (T = 1.018, 0.7% ECE reduction) because the miscalibration is non-uniform across the confidence range rather than a uniform over-confidence. Isotonic regression, fitted on the validation split and evaluated on test, reduced ECE by **91%** (0.155 → 0.014 for actionable predictions).

**The calibrator is applied in the deployed demo.** Displayed confidences reflect measured precision on this dataset. Calibration does not change detection accuracy — mAP is unchanged — it makes the confidence values honest.

## Limitations (evaluated, not assumed)

- **Data scarcity is the binding constraint.** This was tested interventionally: switching to a stronger architecture (YOLOv8s) improved all 14 classes, yet data-scarce classes remained unusable. Architecture is not the limitation.
- **Twelve of fourteen classes should not be relied upon.** Classes with 15–30 training examples produce frequent false positives, visible in the failure analysis.
- **Calibration is dataset-specific.** The isotonic curve is fitted to this data's distribution and would require refitting for other populations or equipment.
- **Attention is diffuse.** Feature-activation mapping shows the model attends to thoracic anatomy rather than artifacts, but broadly rather than focally — consistent with over-prediction.
- **Trained on a subset** due to compute constraints. Full-dataset training would likely improve rare-class performance.
- **Out-of-distribution performance untested.**

## Ethical Considerations

Medical AI carries real risk of harm if misused. This model is deliberately framed as educational only, with disclaimers throughout the demo, API responses, and documentation. Calibrated confidences are shipped specifically so the interface does not overstate certainty. It should not be deployed anywhere its output could influence a health decision.

## Methodology Notes

- Transfer learning from COCO-pretrained weights (both models)
- Class-balanced sampling, rare classes oversampled up to 28× (Faster R-CNN)
- Test-time augmentation (hflip + WBF): mAP@0.5:0.95 0.126 → 0.132
- Best checkpoint by validation loss (epoch 4; overfitting beyond)
- Calibrator stored as a plain JSON curve rather than a pickled model — version-independent and dependency-free
