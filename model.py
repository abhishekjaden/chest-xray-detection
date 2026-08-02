# model.py - YOLOv8s inference with isotonic calibration
import json
import numpy as np
from PIL import Image

# Class names (0-13, no background class in YOLO)
CLASS_NAMES = ['Aortic enlargement','Atelectasis','Calcification','Cardiomegaly',
    'Consolidation','ILD','Infiltration','Lung Opacity','Nodule/Mass','Other lesion',
    'Pleural effusion','Pleural thickening','Pneumothorax','Pulmonary fibrosis']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 512
WEIGHTS = 'yolov8s_best.pt'

# Load the isotonic calibration curve (fitted on held-out validation data).
# Stored as plain numbers: version-independent, no sklearn dependency.
_calib_x, _calib_y = None, None
try:
    with open('calibration_curve_yolo.json') as f:
        _curve = json.load(f)
    _calib_x = np.array(_curve['x'])
    _calib_y = np.array(_curve['y'])
    print(f'Calibration curve loaded ({len(_calib_x)} points).')
except Exception as e:
    print(f'Calibration curve not loaded ({e}); falling back to raw scores.')


def calibrate(raw_score):
    """Map a raw confidence score to a calibrated one via the isotonic curve."""
    if _calib_x is None:
        return raw_score
    return float(np.interp(raw_score, _calib_x, _calib_y))


def read_dicom(file_bytes):
    """Convert DICOM bytes to a PIL RGB image (same preprocessing as training)."""
    import pydicom
    import io
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
    pixel_array = pixel_array.astype(np.uint8)
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = 255 - pixel_array
    return Image.fromarray(pixel_array).convert('RGB')


_model = None  # cached model instance


def load_model():
    """Load the YOLOv8s model. Called once at startup."""
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    _model = YOLO(WEIGHTS)
    return _model


def predict(pil_image, score_threshold=0.25, use_tta=False):
    """Run inference on a PIL image. Returns list of detections.

    Note: use_tta is accepted for interface compatibility but not used —
    YOLOv8 is run single-pass. Calibration was fitted on single-pass scores.
    """
    model = load_model()
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))

    results = model.predict(np.array(img), imgsz=IMG_SIZE,
                            conf=score_threshold, verbose=False)[0]

    detections = []
    for box, score, cls in zip(results.boxes.xyxy.cpu().numpy(),
                               results.boxes.conf.cpu().numpy(),
                               results.boxes.cls.cpu().numpy()):
        raw = float(score)
        detections.append({
            'label': CLASS_NAMES[int(cls)],
            'confidence': round(calibrate(raw), 3),   # calibrated (primary)
            'raw_confidence': round(raw, 3),          # original model score
            'box': [round(float(c), 1) for c in box],
        })
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def draw_detections(pil_image, detections):
    """Draw bounding boxes + labels on the image. Returns annotated PIL image (512x512)."""
    from PIL import ImageDraw, ImageFont
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    palette = [
        '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4',
        '#46f0f0','#f032e6','#bcf60c','#fabebe','#008080','#e6beff',
        '#9a6324','#fffac8'
    ]
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        ci = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        color = palette[ci % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} {conf:.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0], tb[1]-2, tb[2]+4, tb[3]+2], fill=color)
        draw.text((x1+2, y1), text, fill='white', font=font)

    return img
