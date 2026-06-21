# Model Card: Chest X-Ray Abnormality Detection

## Model Overview

A Faster R-CNN (ResNet-50 FPN backbone) object detector that localizes 14 thoracic abnormalities in chest radiographs. Trained on the VinBigData Chest X-ray dataset and deployed as an educational demonstration.

- **Model type:** Object detection (Faster R-CNN)
- **Framework:** PyTorch / torchvision
- **Input:** Chest X-ray (PNG, JPG, or DICOM), resized to 512×512
- **Output:** Bounding boxes with class labels and confidence scores
- **Live demo:** https://huggingface.co/spaces/rocky17435/xray-detection

## Intended Use

**Intended:** Educational and research demonstration of an end-to-end medical-imaging ML system. Illustrative of object-detection methods applied to radiographs.

**NOT intended:** Clinical diagnosis, screening, triage, or any medical decision-making. This model is **not** a medical device, has **no** regulatory clearance, and must **never** be used to inform patient care.

## Training Data

- **Source:** VinBigData Chest X-ray Abnormalities Detection dataset
- **Subset used:** ~3,075 training / 659 validation / 660 test images (stratified split)
- **Labels:** Multi-radiologist annotations merged via Weighted Boxes Fusion (36,096 raw boxes → 22,719 consensus boxes)
- **Known imbalance:** Severe class imbalance (~65:1). Common classes (Aortic enlargement: 974 images) vastly outnumber rare ones (Atelectasis: 15 images).

## Performance

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.29 |
| mAP@0.5:0.95 | 0.13 |

Comparable to published competition baselines for this dataset.

### Per-class (mAP@0.5:0.95) — performance tracks data volume

| Well-represented | Score | Data-scarce | Score |
|------------------|-------|-------------|-------|
| Cardiomegaly | 0.54 | Other lesion | 0.01 |
| Aortic enlargement | 0.51 | Calcification | 0.03 |
| Pleural effusion | 0.10 | Pleural thickening | 0.03 |

## Limitations (evaluated, not assumed)

- **Data scarcity dominates performance.** Rare classes (15–30 training examples) are effectively undetectable. Failure-case analysis visually confirms over-prediction and false positives on these classes.
- **Confidence scores are not calibrated.** Calibration analysis (13,152 predictions) shows the model is well-calibrated at extremes (84% precision at >0.9 confidence) but **over-confident in the mid-range** (~40% precision at 0.75 confidence). Confidence values should not be interpreted as probabilities.
- **Attention is diffuse.** Feature-activation mapping shows the model attends to thoracic anatomy (not artifacts), but attention is broad rather than tightly localized — consistent with the over-prediction tendency.
- **Trained on a subset** of the full dataset due to compute constraints; full-data training would likely improve rare-class performance.
- **Distribution shift:** performance on X-rays outside the VinBigData distribution (different equipment, populations) is untested and likely degraded.

## Ethical Considerations

Medical AI carries risk of harm if misused. This model is deliberately framed as educational only, with prominent disclaimers throughout the demo and API. It should not be deployed in any setting where its output could influence health decisions.

## Methodology Notes

- Transfer learning from COCO-pretrained Faster R-CNN
- Class-balanced sampling (rare classes oversampled up to 28×)
- Test-Time Augmentation (horizontal flip + WBF) at inference: improved mAP@0.5:0.95 from 0.126 to 0.132
- Best checkpoint selected by validation loss (epoch 4; overfitting observed beyond)