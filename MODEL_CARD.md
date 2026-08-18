# Model Card: Chest X-Ray Abnormality Detection

## Model Overview

Object detectors that localize 14 thoracic abnormalities in chest radiographs, trained on the VinBigData Chest X-ray dataset and deployed as an educational demonstration.

- **Task:** Multi-class object detection
- **Models:** YOLOv8s (11.1M params) — **deployed**; Faster R-CNN (ResNet-50 FPN, 41.3M params) — evaluated for comparison
- **Framework:** PyTorch / torchvision / Ultralytics
- **Input:** Chest X-ray (PNG, JPG, or DICOM via the API; PNG/JPG in the Gradio demo), resized to 512×512
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
- **Imbalance:** Severe within the sampled subset, roughly 65:1 — Aortic enlargement in 974 images, Atelectasis in 15. The full training set is less skewed, and imbalance does not predict per-class performance (see Limitations).

## Performance

Both models on the same held-out test split:

| | Faster R-CNN | YOLOv8s (deployed) |
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

Both models were poorly calibrated, in opposite directions.

**Faster R-CNN was over-confident.** Predictions claiming 75% confidence were correct 40% of the time (ECE = 0.155 for scores ≥ 0.25). Temperature scaling was ineffective (T = 1.018, 0.7% ECE reduction) because the miscalibration is non-uniform across the confidence range rather than a uniform inflation — accurate at both extremes, badly wrong in the middle. Isotonic regression, being non-parametric, fit the actual shape and reduced ECE by **91%** (0.155 → 0.014).

**YOLOv8s is under-confident.** Raw scores of 0.35 were correct 62% of the time; 0.83 was correct 99% of the time. Isotonic calibration reduced ECE by **93.7%** (0.249 → 0.016).

**Calibrated output is capped at 0.95.** The uncapped fit mapped high scores to a literal 1.0, but that rested on only 4 test samples in the top confidence bin — not statistically supported, and inappropriate to display as certainty in a medical context. The cap costs 0.0012 ECE.

**Known limitation:** the isotonic fit has flat regions where distinct raw scores collapse to identical calibrated values (raw 0.35 and 0.55 both map to 0.69). This reflects monotonicity enforcement through noisy mid-range bins on 6,754 test predictions.

**The YOLOv8s calibrator is applied in the deployed demo.** Displayed confidences reflect measured precision on this dataset. Calibration does not change detection accuracy — mAP is unchanged — it makes the confidence values honest.

## Limitations (evaluated, not assumed)

- **The cause of the per-class failures is not established.** Two hypotheses were tested and rejected: architecture (YOLOv8s improved all 14 classes without changing which fail) and training-set size (ρ = +0.055, p = 0.85 against per-class mAP). Positional spread of annotations is the strongest remaining candidate but does not reach significance at n = 14.
- **Twelve of fourteen classes should not be relied upon.** They score below 0.18 mAP@0.5:0.95 in both models, independent of how many training examples exist for them.
- **Calibration is dataset-specific.** The isotonic curve is fitted to this data's distribution and would require refitting for other populations or equipment.
- **Conservative on out-of-distribution images.** YOLOv8s produces roughly 3.6× fewer predictions than Faster R-CNN at more than double the precision, and frequently returns no findings on radiographs unlike its training data.
- **Attention is diffuse.** Feature-activation mapping (measured on Faster R-CNN) shows the model attends to thoracic anatomy rather than artifacts, but broadly rather than focally.
- **Trained on a subset** due to compute constraints. Note that full-dataset training would not be expected to fix the failing classes, since per-class performance does not correlate with training-set size.

## Ethical Considerations

Medical AI carries real risk of harm if misused. This model is deliberately framed as educational only, with disclaimers throughout the demo, API responses, and documentation. Calibrated confidences are shipped specifically so the interface does not overstate certainty. It should not be deployed anywhere its output could influence a health decision.

## Methodology Notes

- Transfer learning from COCO-pretrained weights (both models)
- Faster R-CNN: class-balanced sampling (rare classes oversampled up to 28×), best checkpoint by validation loss (epoch 4; overfitting beyond), TTA at inference (hflip + WBF, mAP@0.5:0.95 0.126 → 0.132)
- YOLOv8s: default Ultralytics augmentation, early-stopped at epoch 46 (best at 36), single-pass inference — no TTA
- Calibrators stored as plain JSON curves rather than pickled models — version-independent and dependency-free
