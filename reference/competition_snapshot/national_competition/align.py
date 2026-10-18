"""Reproducible raw-output alignment on one image; OM result is supplied by board runtime."""
import argparse
import json
from pathlib import Path

import numpy as np

from .model_tools import compare_raw, inspect_onnx, inspect_pt, sha256


def preprocess(image_path):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    h, w = image.shape[:2]
    scale = min(640 / h, 640 / w)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    top = (640 - resized.shape[0]) // 2
    bottom = 640 - resized.shape[0] - top
    left = (640 - resized.shape[1]) // 2
    right = 640 - resized.shape[1] - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)[None] / 255.0
    return tensor, {"original_hw": [h, w], "scale": scale,
                    "padding_ltrb": [left, top, right, bottom], "color": "RGB", "range": "0..1"}


def pt_output(pt, tensor):
    import torch
    from ultralytics import YOLO

    model = YOLO(str(pt)).model.eval().cpu()
    with torch.no_grad():
        output = model(torch.from_numpy(tensor))
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().numpy()


def onnx_output(onnx, tensor):
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("Expected one input and one raw output")
        return session.run(None, {session.get_inputs()[0].name: tensor})[0]
    except (ImportError, OSError):
        import cv2
        net = cv2.dnn.readNetFromONNX(str(onnx))
        net.setInput(tensor)
        return net.forward()


def align(pt, onnx, image, output_dir, om_raw=None):
    inspect_pt(pt)
    inspect_onnx(onnx)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor, prep = preprocess(image)
    np.save(output_dir / "input.npy", tensor)
    pt_raw = pt_output(pt, tensor)
    onnx_raw = onnx_output(onnx, tensor)
    np.save(output_dir / "pt_raw.npy", pt_raw)
    np.save(output_dir / "onnx_raw.npy", onnx_raw)
    report = {"pt_sha256": sha256(pt), "onnx_sha256": sha256(onnx),
              "image_sha256": sha256(image), "preprocess": prep,
              "pt_vs_onnx": compare_raw(output_dir / "pt_raw.npy", output_dir / "onnx_raw.npy")}
    if om_raw is not None:
        report["om_raw_sha256"] = sha256(om_raw)
        report["pt_vs_om"] = compare_raw(output_dir / "pt_raw.npy", om_raw)
        report["onnx_vs_om"] = compare_raw(output_dir / "onnx_raw.npy", om_raw)
    (output_dir / "alignment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def decode_raw(path, preprocess_info, conf=0.35, nms_iou=0.55):
    import cv2

    raw = np.load(path).reshape(1, 21, 8400)[0].T
    scores = raw[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= conf
    rows, class_ids, confidences = raw[keep], class_ids[keep], confidences[keep]
    h, w = preprocess_info["original_hw"]
    scale = preprocess_info["scale"]
    left, top, _, _ = preprocess_info["padding_ltrb"]
    boxes_xywh, boxes_xyxy = [], []
    for row in rows:
        cx, cy, bw, bh = row[:4]
        x1 = float(np.clip((cx - bw / 2 - left) / scale, 0, w - 1))
        y1 = float(np.clip((cy - bh / 2 - top) / scale, 0, h - 1))
        x2 = float(np.clip((cx + bw / 2 - left) / scale, 0, w - 1))
        y2 = float(np.clip((cy + bh / 2 - top) / scale, 0, h - 1))
        boxes_xywh.append([x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)])
        boxes_xyxy.append([x1, y1, x2, y2])
    selected = []
    object_positions = np.where(class_ids < 16)[0]
    if len(object_positions):
        indices = cv2.dnn.NMSBoxes([boxes_xywh[i] for i in object_positions],
                                  [float(confidences[i]) for i in object_positions],
                                  conf, nms_iou)
        selected.extend(object_positions[int(i)] for i in np.asarray(indices).reshape(-1))
    table_positions = np.where(class_ids == 16)[0]
    if len(table_positions):
        selected.append(table_positions[int(np.argmax(confidences[table_positions]))])
    return [{"class_id": int(class_ids[i]), "confidence": float(confidences[i]),
             "xyxy": boxes_xyxy[i]} for i in selected]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def compare_decoded(reference, candidate, alignment_json, conf=0.35, nms_iou=0.55):
    info = json.loads(Path(alignment_json).read_text(encoding="utf-8"))["preprocess"]
    expected = decode_raw(reference, info, conf, nms_iou)
    actual = decode_raw(candidate, info, conf, nms_iou)
    remaining = set(range(len(actual)))
    matches = []
    for ref in expected:
        possible = [(box_iou(ref["xyxy"], actual[i]["xyxy"]), i) for i in remaining
                    if actual[i]["class_id"] == ref["class_id"]]
        if not possible:
            continue
        overlap, index = max(possible)
        remaining.remove(index)
        matches.append({"class_id": ref["class_id"], "iou": overlap,
                        "confidence_abs": abs(ref["confidence"]-actual[index]["confidence"])})
    return {"reference_count": len(expected), "candidate_count": len(actual),
            "reference_classes": sorted(x["class_id"] for x in expected),
            "candidate_classes": sorted(x["class_id"] for x in actual),
            "matched": len(matches), "unmatched_candidate": len(remaining),
            "mean_iou": sum(x["iou"] for x in matches)/max(1, len(matches)),
            "min_iou": min((x["iou"] for x in matches), default=0.0),
            "max_confidence_abs": max((x["confidence_abs"] for x in matches), default=0.0),
            "matches": matches}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--om-raw", type=Path)
    args = parser.parse_args()
    result = align(args.pt, args.onnx, args.image, args.output_dir, args.om_raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))
