# model.py - loads the trained model and runs TTA inference
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image

# Load config
with open('inference_config.json') as f:
    CONFIG = json.load(f)

CLASS_NAMES = CONFIG['class_names']
NUM_CLASSES = CONFIG['num_classes']
IMG_SIZE = CONFIG['img_size']
MEAN = CONFIG['mean']
STD = CONFIG['std']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    """Load the Faster R-CNN with trained weights. Called once at startup."""
    global _model
    if _model is not None:
        return _model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    _model = model
    return model


def preprocess(pil_image):
    """Resize to 512, normalize, return tensor."""
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, mean=MEAN, std=STD)
    return tensor


def predict(pil_image, score_threshold=0.25, use_tta=True):
    """Run TTA inference on a PIL image. Returns list of detections."""
    model = load_model()
    tensor = preprocess(pil_image)

    with torch.no_grad():
        out_orig = model([tensor.to(DEVICE)])[0]

    if use_tta:
        flipped = torch.flip(tensor, dims=[2])
        with torch.no_grad():
            out_flip = model([flipped.to(DEVICE)])[0]

        fb = out_flip['boxes'].cpu().numpy().copy()
        fb_un = fb.copy()
        fb_un[:, 0] = IMG_SIZE - fb[:, 2]
        fb_un[:, 2] = IMG_SIZE - fb[:, 0]

        bo = np.clip(out_orig['boxes'].cpu().numpy() / IMG_SIZE, 0, 1)
        bf = np.clip(fb_un / IMG_SIZE, 0, 1)
        so = out_orig['scores'].cpu().numpy()
        sf = out_flip['scores'].cpu().numpy()
        lo = out_orig['labels'].cpu().numpy()
        lf = out_flip['labels'].cpu().numpy()

        if len(bo) == 0 and len(bf) == 0:
            return []

        boxes, scores, labels = weighted_boxes_fusion(
            [bo.tolist(), bf.tolist()], [so.tolist(), sf.tolist()],
            [lo.tolist(), lf.tolist()], iou_thr=0.5, skip_box_thr=0.0)
        boxes = np.array(boxes) * IMG_SIZE
    else:
        boxes = out_orig['boxes'].cpu().numpy()
        scores = out_orig['scores'].cpu().numpy()
        labels = out_orig['labels'].cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            detections.append({
                'label': CLASS_NAMES[int(label)],
                'confidence': round(float(score), 3),
                'box': [round(float(c), 1) for c in box],  # [x1,y1,x2,y2] in 512-space
            })
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections
def draw_detections(pil_image, detections):
    """Draw bounding boxes + labels on the image. Returns annotated PIL image (512x512)."""
    from PIL import ImageDraw, ImageFont
    # Resize to the processing size so boxes (in 512-space) align
    img = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)

    # A distinct color per class index
    palette = [
        '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4',
        '#46f0f0','#f032e6','#bcf60c','#fabebe','#008080','#e6beff',
        '#9a6324','#fffac8','#800000'
    ]
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        # color by class name index
        ci = CLASS_NAMES.index(label) if label in CLASS_NAMES else 0
        color = palette[ci % len(palette)]
        # box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # label background + text
        text = f"{label} {conf:.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0], tb[1]-2, tb[2]+4, tb[3]+2], fill=color)
        draw.text((x1+2, y1), text, fill='white', font=font)

    return img