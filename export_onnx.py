"""
Export the trained YOLOv8s detector to ONNX and verify parity with ultralytics.

The verification is the point: ONNX post-processing (box decoding + NMS) is
reimplemented here, so it must be checked against the reference implementation
rather than assumed correct. Results in analysis/onnx_evaluation.json.

Usage:  python export_onnx.py --weights yolov8s_best.onnx --images path/to/test/images
"""
import argparse
import numpy as np
from PIL import Image

IMG_SIZE = 512


def export(weights_pt, imgsz=IMG_SIZE, opset=12):
    """Export a .pt checkpoint to ONNX. Returns the output path."""
    from ultralytics import YOLO
    model = YOLO(weights_pt)
    return model.export(format='onnx', imgsz=imgsz, opset=opset, simplify=True)


def nms(boxes, scores, iou_thr=0.7):
    """Non-maximum suppression. boxes in xyxy, returns kept indices."""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = ((boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) *
                  (boxes[idxs[1:], 3] - boxes[idxs[1:], 1]))
        iou = inter / (area_i + area_o - inter + 1e-9)
        idxs = idxs[1:][iou < iou_thr]
    return keep


def onnx_predict(session, input_name, pil_img, conf=0.25, iou_thr=0.7):
    """Run inference and post-process. Returns [(class_id, score, xyxy), ...].

    YOLOv8 is anchor-free: raw output is (1, 4+num_classes, num_anchors) with
    boxes in centre-xywh and per-class scores (no separate objectness term).
    """
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]                        # HWC -> NCHW

    out = session.run(None, {input_name: x})[0][0].T      # -> (num_anchors, 4+nc)
    cxcywh, cls_scores = out[:, :4], out[:, 4:]

    best_cls = cls_scores.argmax(1)
    best_score = cls_scores.max(1)
    mask = best_score >= conf
    cxcywh, best_cls, best_score = cxcywh[mask], best_cls[mask], best_score[mask]
    if len(cxcywh) == 0:
        return []

    xyxy = np.stack([
        cxcywh[:, 0] - cxcywh[:, 2] / 2,
        cxcywh[:, 1] - cxcywh[:, 3] / 2,
        cxcywh[:, 0] + cxcywh[:, 2] / 2,
        cxcywh[:, 1] + cxcywh[:, 3] / 2,
    ], axis=1)

    results = []
    for c in np.unique(best_cls):                          # NMS per class
        ci = np.where(best_cls == c)[0]
        for k in nms(xyxy[ci], best_score[ci], iou_thr):
            j = ci[k]
            results.append((int(c), float(best_score[j]), xyxy[j]))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def verify(onnx_path, pt_path, image_paths, conf=0.25):
    """Compare ONNX output against ultralytics on the same images."""
    import onnxruntime as ort
    from ultralytics import YOLO

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    reference = YOLO(pt_path)

    all_ok = True
    for path in image_paths:
        ref = reference.predict(path, imgsz=IMG_SIZE, conf=conf, verbose=False)[0]
        ref_res = sorted(
            [(int(c), float(s), b) for b, s, c in zip(
                ref.boxes.xyxy.cpu().numpy(),
                ref.boxes.conf.cpu().numpy(),
                ref.boxes.cls.cpu().numpy())],
            key=lambda r: r[1], reverse=True)
        onnx_res = onnx_predict(session, input_name, Image.open(path), conf=conf)

        if len(ref_res) != len(onnx_res):
            print(f'{path}: COUNT MISMATCH {len(ref_res)} vs {len(onnx_res)}')
            all_ok = False
            continue
        for (c1, s1, b1), (c2, s2, b2) in zip(ref_res, onnx_res):
            dbox = float(np.abs(np.array(b1) - np.array(b2)).max())
            ok = c1 == c2 and abs(s1 - s2) < 0.01 and dbox < 2.0
            all_ok &= ok
            print(f'{path.split("/")[-1][:16]}  cls {c1}/{c2}  '
                  f'score {s1:.3f}/{s2:.3f}  box Δ {dbox:.2f}px  '
                  f'{"OK" if ok else "*** MISMATCH"}')
    return all_ok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pt', default='yolov8s_best.pt', help='PyTorch checkpoint')
    p.add_argument('--onnx', default=None, help='existing ONNX file; exports if omitted')
    p.add_argument('--images', nargs='+', required=True, help='images to verify against')
    args = p.parse_args()

    onnx_path = args.onnx or export(args.pt)
    print(f'ONNX: {onnx_path}\n')
    ok = verify(onnx_path, args.pt, args.images)
    print(f'\nParity: {"PASS" if ok else "FAIL"}')
