#!/usr/bin/env python3
"""Ascend worker for national_A_20260927 with unified thresholds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
import cv2
import numpy as np
from national_competition.runtime.depth_validation import rgb_for_sequence, capture_active
from national_competition.runtime.water_dedup import suppress_water_capacity_duplicates
from national_competition.runtime.ocr_source import read_ocr_source
from national_competition.runtime.native_yolo_source import canonical_native_view,compatibility_view,project_detections,INPUT_POLICY
from national_competition.runtime.capture_policy import read_capture_policy
from national_competition.config import CLASS_CONF_OVERRIDES

ACL_LITE_ROOT = "/usr/local/Ascend/thirdpart/aarch64"
sys.path.insert(0, ACL_LITE_ROOT)
sys.path.insert(0, os.path.join(ACL_LITE_ROOT, "acllite"))
from acllite.acllite_model import AclLiteModel
from acllite.acllite_resource import AclLiteResource

MODEL = Path.home() / "national_competition/models/national_A_20260927.om"


def letterbox(image, size=640):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) / 2, (size - nh) / 2
    left, top = int(round(px - 0.1)), int(round(py - 0.1))
    right, bottom = size - nw - left, size - nh - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32) / 255.0, scale, left, top


def decode(output, shape, scale, px, py, object_conf=0.35, table_conf=0.25, iou=0.55, include_capacity_scores=False):
    pred = np.asarray(output).reshape(1, 21, 8400)[0].T
    scores = pred[:, 4:]
    ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), ids]
    thresholds = np.where(ids == 16, table_conf, object_conf)
    for cls, threshold in CLASS_CONF_OVERRIDES.items():
        thresholds[ids == cls] = threshold
    keep = (confs >= thresholds) & (ids <= 16)
    pred, ids, confs = pred[keep], ids[keep], confs[keep]
    h, w = shape[:2]
    boxes, xyxy = [], []
    for row in pred:
        cx, cy, bw, bh = row[:4]
        x1 = float(np.clip((cx - bw / 2 - px) / scale, 0, w - 1))
        y1 = float(np.clip((cy - bh / 2 - py) / scale, 0, h - 1))
        x2 = float(np.clip((cx + bw / 2 - px) / scale, 0, w - 1))
        y2 = float(np.clip((cy + bh / 2 - py) / scale, 0, h - 1))
        boxes.append([x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)])
        xyxy.append([x1, y1, x2, y2])
    result = []
    object_positions = np.where(ids < 16)[0]
    if len(object_positions):
        selected = cv2.dnn.NMSBoxes([boxes[i] for i in object_positions],
                                    [float(confs[i]) for i in object_positions], 0.0, iou)
        for local in np.asarray(selected).reshape(-1) if len(selected) else []:
            index = object_positions[int(local)]
            detection = (int(ids[index]), float(confs[index]), xyxy[index])
            if include_capacity_scores and ids[index] in (10, 11):
                detection += (tuple(float(v) for v in pred[index, 14:16]),)
            result.append(detection)
    table_positions = np.where(ids == 16)[0]
    if len(table_positions):
        index = table_positions[int(np.argmax(confs[table_positions]))]
        result.append((16, float(confs[index]), xyxy[index]))
    return suppress_water_capacity_duplicates(result)


def atomic_write(path, sequence, detections):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="ascii") as stream:
        stream.write(f"{sequence}\n")
        for detection in detections:
            class_id, confidence, box = detection[:3]
            capacity = (',' + ','.join(f'{v:.6f}' for v in detection[3])) if len(detection)>3 else ''
            stream.write(f"{class_id},{confidence:.6f}," +
                         ",".join(f"{value:.2f}" for value in box) + capacity + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("square", "round"), default="square")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--ready", type=Path)
    args = parser.parse_args()
    prefix = "round_yolo" if args.mode == "round" else "astra_yolo"
    frame = args.frame or Path(f"/dev/shm/{prefix}_frame.jpg")
    request = args.request or Path(f"/dev/shm/{prefix}_request.txt")
    result = args.result or Path(f"/dev/shm/{prefix}_result.txt")
    ready = args.ready or Path(f"/dev/shm/national_A_20260927_{args.mode}.ready")
    ready.unlink(missing_ok=True)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    print(f"STARTUP imports={time.monotonic()-STARTED:.3f}s", flush=True)
    phase = time.monotonic()
    resource = AclLiteResource()
    resource.init()
    print(f"STARTUP acl_init={time.monotonic()-phase:.3f}s", flush=True)
    phase = time.monotonic()
    model = AclLiteModel(str(args.model))
    print(f"STARTUP model_load={time.monotonic()-phase:.3f}s", flush=True)
    phase = time.monotonic()
    model.execute([np.zeros((1, 3, 640, 640), dtype=np.float32)])
    print(f"STARTUP first_inference={time.monotonic()-phase:.3f}s total={time.monotonic()-STARTED:.3f}s", flush=True)
    settings_path=Path.home()/'national_competition/ocr/ocr_rules.json'
    require_native=json.loads(settings_path.read_text(encoding='utf-8')).get('high_resolution_rgb_enabled',False) if settings_path.exists() else False
    def publish_ready():
        temporary = ready.with_suffix(".ready.tmp")
        temporary.write_text(
            f"pid={os.getpid()} model={args.model} time={time.time_ns()} input_policy={INPUT_POLICY}\n", encoding="ascii")
        os.replace(temporary, ready)

    publish_ready()
    last_ready_refresh = time.monotonic()
    last_sequence = -1
    print(f"National worker ready: mode={args.mode} model={args.model}", flush=True)
    while True:
        if not ready.exists() or time.monotonic() - last_ready_refresh >= 5.0:
            publish_ready()
            last_ready_refresh = time.monotonic()
        try:
            sequence = int(request.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError, OSError):
            time.sleep(0.005)
            continue
        if sequence == last_sequence:
            time.sleep(0.003)
            continue
        paired = capture_active() and args.mode == 'square'
        mapping=None
        capture_policy=read_capture_policy(sequence) if paired else None
        input_policy='reference-rgb-legacy'
        if paired and capture_policy and capture_policy['compatibility_mode']:
            try:image,mapping,input_policy=compatibility_view(rgb_for_sequence(sequence))
            except ValueError:
                time.sleep(.005)
                continue
        elif paired and require_native:
            high,source_to_reference,source_status=read_ocr_source(sequence)
            try:image,mapping=canonical_native_view(high,source_to_reference)
            except ValueError:
                time.sleep(.005)
                continue
            input_policy=INPUT_POLICY
        else:
            image = rgb_for_sequence(sequence) if paired else cv2.imread(str(frame))
        if image is None:
            time.sleep(0.005)
            continue
        tensor, scale, px, py = letterbox(image)
        output = model.execute([tensor])[0]
        detections=decode(output,image.shape,scale,px,py,include_capacity_scores=True)
        if mapping is not None:detections=project_detections(detections,mapping)
        policy={'sequence':sequence,'input_policy':input_policy,'capture_policy':capture_policy,
                'input_shape':[int(image.shape[1]),int(image.shape[0])],
                'canonical_to_reference':mapping.tolist() if mapping is not None else None}
        policy_path=result.with_suffix('.input.json');temporary=policy_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(policy),encoding='ascii');os.replace(temporary,policy_path)
        atomic_write(result,sequence,detections)
        last_sequence = sequence


if __name__ == "__main__":
    main()
