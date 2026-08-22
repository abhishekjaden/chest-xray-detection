# Technical Report: Chest X-Ray Abnormality Detection

An end-to-end study of object detection for thoracic abnormalities in chest radiographs, covering model development, deployment, and — the main contribution — an investigation into *why* per-class performance varies as it does.

> **Educational / research demonstration only. Not a diagnostic device. Not clinically validated.**

**Live demo:** https://huggingface.co/spaces/rocky17435/xray-detection (YOLOv8s, isotonic-calibrated)

---

## 1. Problem and Data

Multi-class object detection over 14 thoracic findings, using the VinDr-CXR / VinBigData chest X-ray dataset.

**Label fusion.** Each image carries annotations from multiple radiologists, who frequently disagree on both presence and extent of findings. Rather than treating each annotation as independent ground truth, overlapping boxes were merged with Weighted Boxes Fusion, reducing 36,096 raw annotations to 22,719 consensus boxes (a 37% reduction).

**Splits.** Stratified into 3,075 train / 659 validation / 660 test images. Results are on the held-out test split unless stated otherwise.

**Class imbalance.** Roughly 65:1 within the sampled subset — Aortic enlargement in 974 images, Atelectasis in 15. This appeared to be the central constraint, and Section 3 tests that assumption against the full dataset.

---

## 2. Model Development

Two detectors were trained on identical data and splits.

**Faster R-CNN** (ResNet-50 FPN, 41.3M params), COCO-pretrained. Class-balanced sampling oversampled rare-finding images up to 28×. Validation loss bottomed at epoch 4 and rose thereafter while training loss kept falling — clear overfitting, so the epoch-4 checkpoint was selected. Test-time augmentation (horizontal flip + WBF fusion) improved mAP@0.5:0.95 from 0.126 to 0.132.

**YOLOv8s** (11.1M params), COCO-pretrained, same 512px inputs and splits. Trained 46 epochs with early stopping (best at 36), default Ultralytics augmentation, single-pass inference.

| | Faster R-CNN | YOLOv8s |
|---|---|---|
| Parameters | 41.3M | 11.1M |
| mAP@0.5 | 0.290 | **0.348** |
| mAP@0.5:0.95 | 0.132 | **0.180** |
| Inference | two-pass + WBF | ~7 ms/image |

**YOLOv8s is the deployed model.**

---

## 3. Per-Class Analysis

Overall mAP conceals a two-tier structure (YOLOv8s, mAP@0.5:0.95):

| Class | Images (full train set) | Median box area (px²) | Positional spread (px) | mAP |
|---|---|---|---|---|
| Cardiomegaly | 2,316 | 396,385 | 322 | 0.655 |
| Aortic enlargement | 3,098 | 96,503 | 277 | 0.590 |
| Pleural effusion | 1,038 | 62,367 | 925 | 0.170 |
| Pneumothorax | 96 | 530,909 | 808 | 0.150 |
| Nodule/Mass | 841 | 10,012 | 803 | 0.149 |
| Consolidation | 353 | 235,992 | 658 | 0.142 |
| ILD | 397 | 548,439 | 715 | 0.122 |
| Atelectasis | 187 | 272,439 | 704 | 0.117 |
| Infiltration | 613 | 244,912 | 730 | 0.105 |
| Pulmonary fibrosis | 1,621 | 75,729 | 761 | 0.099 |
| Lung Opacity | 1,331 | 151,715 | 742 | 0.097 |
| Pleural thickening | 2,010 | 31,330 | 1,058 | 0.063 |
| Calcification | 458 | 27,206 | 764 | 0.036 |
| Other lesion | 1,154 | 119,592 | 852 | 0.029 |

**An initial hypothesis — that performance is limited by training-set size — was tested against the full VinDr-CXR annotations and does not hold.**

| Variable vs mAP@0.5:0.95 | Spearman ρ | p |
|---|---|---|
| Training images | +0.055 | 0.85 |
| Median box area | +0.292 | 0.31 |
| Positional spread | −0.433 | 0.12 |
| Boxes per image | +0.233 | 0.42 |

Training-set size shows no relationship with performance. The clearest counter-example: Pleural thickening has 2,010 training images and scores 0.063, while Cardiomegaly has 2,316 and scores 0.655 — near-identical volume, tenfold difference. Pneumothorax scores 0.150 on 96 images, the second-fewest in the dataset.

Section 8 identifies what does predict performance.

---

## 4. Evidence 1 — Qualitative Failure Analysis

![Failure gallery](analysis/failure_analysis_gallery.png)

Ground truth (green) beside predictions (red) across test images shows a consistent pattern:

- **Well-represented classes are detected accurately and localised tightly.** Cardiomegaly and Aortic enlargement predictions closely match ground truth.
- **The persistently failing classes produce false positives.** On difficult images the model floods the field with low-confidence boxes for Pneumothorax, Other lesion, and Pleural thickening — findings it detects poorly regardless of how many training examples exist for them.

---

## 5. Evidence 2 — Confidence Calibration

Both models were poorly calibrated — in opposite directions.

**Method.** Every prediction above 0.05 was scored for correctness (IoU > 0.4 against a same-class ground-truth box) and binned by confidence. Calibrators were fitted on validation and evaluated on test.

### Faster R-CNN: over-confident

24,457 test predictions. ECE was 0.046 overall — but misleading, because 73% of predictions fall below 0.2 confidence where the model happens to be well calibrated. Restricted to actionable predictions (≥ 0.25), **ECE was 0.155**: a prediction claiming 75% confidence was correct only 40% of the time.

**Temperature scaling failed.** T = 1.018, a 0.7% ECE reduction. The reason is diagnostic: temperature scaling applies a single monotonic transform, but the miscalibration is *non-uniform* — accurate at both extremes, badly over-confident in the middle.

**Isotonic regression worked.**

| | ECE (all) | ECE (score ≥ 0.25) |
|---|---|---|
| Raw | 0.0460 | 0.1554 |
| Temperature scaling | 0.0457 | — |
| **Isotonic** | **0.0037** | **0.0140** |

![Reliability diagram](analysis/calibration_reliability_diagram.png)

### YOLOv8s: under-confident

6,754 test predictions — 3.6× fewer than Faster R-CNN, at more than double the precision (0.458 vs ~0.20).

Its scores understate its accuracy at every level: raw 0.07 was correct 32% of the time, raw 0.35 was correct 62%, raw 0.83 was correct 99%. Raw ECE 0.249; isotonic calibration reduced it by **93.7% to 0.016**.

![YOLOv8s reliability diagram](analysis/calibration_yolo_reliability.png)

The curve sits *above* the diagonal — the visual signature of under-confidence, mirroring Faster R-CNN's sag below it. Same fix, opposite direction.

**Output is capped at 0.95.** The uncapped fit mapped high scores to a literal 1.0, resting on 4 test samples in the top bin — not statistically supported, and inappropriate to display as certainty in a medical context. The cap costs 0.0012 ECE.

**Known limitation:** the fit has flat regions where distinct raw scores collapse to the same calibrated value (0.35 and 0.55 both → 0.69), from monotonicity enforcement through noisy mid-range bins.

**Deployed.** The YOLOv8s calibrator is live, stored as a plain JSON curve (`np.interp` lookup) rather than a pickled model — version-independent after an sklearn version warning showed that fragility was real. Calibration does not improve detection accuracy; it makes the confidence numbers honest.

---

## 6. Evidence 3 — Spatial Attention

![Attention heatmap](analysis/gradcam_explainability.png)

Backbone feature-activation mapping (measured on Faster R-CNN) shows the model attends to thoracic anatomy — cardiac silhouette and lung fields — rather than image borders or artifacts. Attention is *diffuse* rather than sharply localised, consistent with the over-prediction seen in the failure analysis.

---

## 7. Evidence 4 — Architecture Comparison

The first interventional test: change the architecture and see whether the limitation persists.

**Setup.** Identical data, splits, 512px inputs, and metric, on the same 660 test images.

![Architecture comparison](analysis/architecture_comparison.png)

YOLOv8s outperforms Faster R-CNN **on all 14 classes** with roughly a quarter of the parameters.

**But the limitation persists.** Absolute gains concentrate in the already-working classes (Cardiomegaly +0.114, Aortic enlargement +0.083), while the failing classes remain unusable: Calcification 0.036, Other lesion 0.029.

**Conclusion:** a substantially better architecture lifts performance across the board but does not change which classes fail. Architecture is not the binding constraint, and Section 3 shows data volume isn't either.

*Caveat: the two models were trained with their conventional recipes (different epoch counts, augmentation stacks, and TTA), so this compares architectures as normally trained rather than isolating architecture alone.*

---

## 8. Evidence 5 — Inter-Radiologist Agreement

Sections 3 and 7 rejected training-set size and architecture. This section tests a third explanation, and finds it holds.

**Hypothesis.** If radiologists disagree about *where* a finding is, the training target is inconsistent and the model cannot learn a stable localisation. Performance would then be bounded by annotation agreement rather than by model capacity or data volume.

**Method.** Each training image was annotated independently by three radiologists. For every (image, class) pair with at least two annotators, the best-matching box pair between each pair of radiologists was found and its IoU recorded, then averaged per class. This measures localisation agreement *given* presence — it does not capture disagreement about whether a finding is there at all.

| Class | Mean pairwise IoU | Cases | mAP@0.5:0.95 |
|---|---|---|---|
| Cardiomegaly | 0.731 | 1,817 | 0.655 |
| Pneumothorax | 0.701 | 58 | 0.150 |
| Aortic enlargement | 0.683 | 2,346 | 0.590 |
| Nodule/Mass | 0.653 | 409 | 0.149 |
| Consolidation | 0.609 | 121 | 0.142 |
| ILD | 0.601 | 152 | 0.122 |
| Infiltration | 0.566 | 245 | 0.105 |
| Calcification | 0.545 | 177 | 0.036 |
| Pulmonary fibrosis | 0.517 | 1,017 | 0.099 |
| Lung Opacity | 0.507 | 547 | 0.097 |
| Pleural effusion | 0.505 | 634 | 0.170 |
| Atelectasis | 0.485 | 62 | 0.117 |
| Other lesion | 0.476 | 362 | 0.029 |
| Pleural thickening | 0.443 | 882 | 0.063 |

![Agreement vs performance](analysis/agreement_vs_map.png)

**Result: ρ = +0.727, p = 0.0032.** Agreement is by a wide margin the strongest predictor tested:

| Variable | Spearman ρ | p |
|---|---|---|
| **Inter-radiologist agreement** | **+0.727** | **0.0032** |
| Positional spread | −0.433 | 0.1220 |
| Median box area | +0.292 | 0.3105 |
| Training-set size | +0.055 | 0.8520 |

**Robustness.** The relationship survives every check applied:

- Controlling for box area: ρ = +0.698. Controlling for training-set size: ρ = +0.734 — essentially unchanged.
- Box area controlling for agreement collapses to ρ = +0.064, indicating lesion size was tracking agreement rather than predicting performance independently.
- Leave-one-out across all 14 classes: ρ ranges +0.659 to +0.890, significant in every case.
- Excluding both top performers, so mAP spans only 0.03–0.17: ρ = +0.580, p = 0.048. The relationship holds *within* the failing classes.

### Independent confirmation

The analysis above has a circularity problem: the evaluation ground truth is a WBF merge of the same annotations from which agreement was computed. Low agreement produces both a noisier training target and a noisier metric, so part of the correlation could reflect measurement unreliability.

To break this, the model was re-evaluated against the **official VinDr-CXR consensus test set** — 300 images sampled for class coverage, annotated by five radiologists with two senior reviewers resolving disagreements. These annotations share no radiologists or process with the training set.

| | Agreement vs mAP | p |
|---|---|---|
| WBF-merged split (circular) | +0.727 | 0.0032 |
| **Consensus test set (independent)** | **+0.802** | **0.0006** |

The correlation is **stronger** against independent labels, not weaker. Class rankings are also stable across the two evaluations (ρ = +0.705, p = 0.005), so the difference between them is a level shift rather than a reordering.

**Conclusion.** Detection performance on this dataset is predicted by inter-radiologist localisation agreement, and not by training-set size, lesion area, or architecture. The practical implication is that reported per-class numbers on low-agreement findings measure annotation consistency as much as model capability: a detector scoring 0.06 on Pleural thickening may be near the ceiling the labels permit.

**Caveats.** n = 14 classes limits statistical power. The consensus evaluation used a 300-image subsample enriched for abnormal cases, so its absolute mAP (0.063 mAP@0.5:0.95) is not comparable to a full-test-set benchmark figure. Consensus boxes also follow a different annotation convention from WBF-merged training boxes, which likely accounts for part of the absolute drop — Aortic enlargement falls from 0.590 to 0.066 with recall 0.545 but precision 0.217, consistent with correct detection under a different boxing convention rather than outright failure.

---

## 9. Engineering

**Serving.** The deployed demo runs YOLOv8s on Hugging Face Spaces via Gradio, accepting PNG, JPG, and DICOM. A FastAPI service provides JSON detections, an annotated-image endpoint, DICOM support, and input validation with size limits and graceful error handling. A React frontend adds client-side threshold filtering and per-class toggles.

**Model swap.** Moving from Faster R-CNN to YOLOv8s required refitting calibration against the new score distribution — the previous curve was fitted to a different model and would have produced wrong confidences. The inference engine was rewritten while keeping the public interface identical (`predict`, `draw_detections`, `read_dicom`), so the API, tests, and demo needed no changes. The existing test suite served as the regression check.

**ONNX export — evaluated, not deployed.** The model was exported to ONNX (opset 12, fixed 512×512 input) with post-processing — anchor-free box decoding and per-class NMS — reimplemented from the raw `(1, 18, 5376)` output. Verified against ultralytics on test images: identical classes, identical scores, **0.00px maximum box difference**. Benchmarked at 163.9 ms/image versus 186.7 ms for ultralytics on CPU — a 1.14× speedup.

It was not deployed. The speedup is immaterial for single-image interactive use, the ONNX file is larger (42.6 MB vs 22.5 MB), and replacing library-maintained post-processing with hand-written decoding introduces maintenance risk against a calibration curve fitted to ultralytics' NMS behaviour. The export script and verification results are in the repository (`export_onnx.py`, `analysis/onnx_evaluation.json`).

**Practices.** 7 endpoint tests covering happy paths and error cases, running in CI on every push; calibration validated on held-out data before deployment; version-independent artifact storage; honest scoping throughout.

---

## 10. Limitations

- Trained on a 3,075-image subset of a 15,000-image training set, which under-represents rare classes relative to the full data (roughly 30 Calcification examples against 458 available; 15 Atelectasis against 187). Correcting this would not be expected to help, given that training-set size shows no correlation with per-class performance.
- Twelve of fourteen classes are unreliable, and the evidence indicates this reflects annotation agreement rather than a fixable model or data deficiency.
- Not clinically validated; no regulatory clearance. Educational demonstration only.
- Calibration is fitted to this dataset's distribution and would require refitting for other data.
- The deployed model is conservative on out-of-distribution images and frequently returns no findings on radiographs unlike its training data.
- The agreement analysis covers 14 classes; n is small and the consensus evaluation used a 300-image subsample.

---

## 11. What Would Actually Help

Three hypotheses were tested and rejected; one held.

**Rejected: architecture.** YOLOv8s outperformed Faster R-CNN on all 14 classes with a quarter of the parameters, without changing which classes fail.

**Rejected: training-set size.** ρ = +0.055 (p = 0.85) against per-class mAP.

**Rejected: input resolution.** Native images are 3072×3072, a 6× downsample to 512px. A median Calcification box becomes roughly 27×27 pixels at input scale — above the 32px COCO "small object" threshold and well within YOLOv8's finest detection stride of 8px. Resolution is unlikely to be limiting for these classes.

**Held: inter-radiologist agreement.** ρ = +0.727 against the WBF split, +0.802 against independent consensus labels, robust to partial correlation and leave-one-out.

Given that, the useful directions are not more data or bigger models:

- **Report agreement alongside per-class metrics.** A benchmark number on a low-agreement class conflates model capability with label consistency, and the two should be separable.
- **Model annotation uncertainty explicitly** rather than collapsing radiologists into a single consensus box — soft targets or per-annotator supervision.
- **Establish an agreement-based ceiling.** Treating each radiologist's annotation as a prediction against the others gives a human-level mAP per class, showing how much headroom actually remains.
- **Extend to all 22 local labels** to raise n beyond 14, which is the binding constraint on statistical power here.
