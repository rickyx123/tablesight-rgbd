"""Runtime-independent filtering and conservative RGB-D evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import median
from typing import Iterable

from .config import DEFAULTS, TABLE_CLASS_ID, WATER_CLASS_IDS, Thresholds


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


class DepthState(str, Enum):
    PASS = "PASS"
    FLAT = "FLAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DepthEvidence:
    state: DepthState
    reason: str
    valid_fraction: float = 0.0
    height_mm: float | None = None


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    overlap = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - overlap
    return overlap / union if union else 0.0


def competing_nms(detections: Iterable[Detection], cfg: Thresholds = DEFAULTS):
    """Suppress only near-identical competing boxes; preserve containment/stacking."""
    kept = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if det.class_id == TABLE_CLASS_ID:
            kept.append(det)
            continue
        if any(old.class_id != TABLE_CLASS_ID
               and iou(det.xyxy, old.xyxy) >= cfg.competition_iou
               and min(area(det.xyxy), area(old.xyxy)) / max(area(det.xyxy), area(old.xyxy), 1) >= cfg.competition_area_ratio
               for old in kept):
            continue
        kept.append(det)
    return kept


def table_filter(detections: Iterable[Detection], cfg: Thresholds = DEFAULTS):
    """Return accepted objects, rejected objects, and diagnostic reason."""
    detections = list(detections)
    tables = [d for d in detections if d.class_id == TABLE_CLASS_ID and d.confidence >= cfg.table_conf]
    objects = [d for d in detections if d.class_id != TABLE_CLASS_ID and d.confidence >= cfg.object_conf]
    if not tables:
        return [], objects, "TABLE_MISSING"
    table = max(tables, key=lambda d: (d.confidence, area(d.xyxy)))
    tx1, ty1, tx2, ty2 = table.xyxy
    mx = cfg.table_margin_fraction * (tx2 - tx1)
    my = cfg.table_margin_fraction * (ty2 - ty1)
    accepted, rejected = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj.xyxy
        px, py = (x1 + x2) / 2, y1 + 0.7 * (y2 - y1)
        (accepted if tx1 - mx <= px <= tx2 + mx and ty1 - my <= py <= ty2 + my else rejected).append(obj)
    return accepted, rejected, "OK"


def depth_evidence(depth_mm, box, table_box, other_boxes=(), cfg: Thresholds = DEFAULTS):
    """Use valid local ring as a support estimate. Depth must be RGB aligned, in mm."""
    import numpy as np

    arr = np.asarray(depth_mm)
    if arr.ndim != 2 or arr.size == 0:
        return DepthEvidence(DepthState.UNKNOWN, "DEPTH_MISSING")
    h, w = arr.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return DepthEvidence(DepthState.UNKNOWN, "EMPTY_BOX")
    for other in other_boxes:
        if iou(box, other) > 0.15:
            return DepthEvidence(DepthState.UNKNOWN, "POSSIBLE_STACK_OR_OCCLUSION")
    inner = arr[y1:y2, x1:x2]
    valid = np.isfinite(inner) & (inner > 100) & (inner < 10000)
    fraction = float(valid.mean())
    if fraction < cfg.depth_min_valid_fraction:
        return DepthEvidence(DepthState.UNKNOWN, "INSUFFICIENT_OBJECT_DEPTH", fraction)
    bw, bh = x2 - x1, y2 - y1
    pad = max(4, int(0.25 * max(bw, bh)))
    rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)
    yy, xx = np.mgrid[ry1:ry2, rx1:rx2]
    tx1, ty1, tx2, ty2 = table_box
    ring = ((xx < x1) | (xx >= x2) | (yy < y1) | (yy >= y2)) & (xx >= tx1) & (xx < tx2) & (yy >= ty1) & (yy < ty2)
    for other in other_boxes:
        ox1, oy1, ox2, oy2 = other
        ring &= ~((xx >= ox1) & (xx < ox2) & (yy >= oy1) & (yy < oy2))
    surround = arr[ry1:ry2, rx1:rx2]
    support = surround[ring & np.isfinite(surround) & (surround > 100) & (surround < 10000)]
    if support.size < cfg.depth_min_ring_pixels:
        return DepthEvidence(DepthState.UNKNOWN, "UNRELIABLE_SUPPORT", fraction)
    support_mm = float(np.median(support))
    if float(np.percentile(np.abs(support - support_mm), 75)) > 15:
        return DepthEvidence(DepthState.UNKNOWN, "NONPLANAR_SUPPORT", fraction)
    heights = support_mm - inner[valid]
    prominent = float(np.percentile(heights, 75))
    positive_fraction = float((heights > cfg.depth_pass_mm).mean())
    if prominent >= cfg.depth_pass_mm and positive_fraction >= 0.2:
        return DepthEvidence(DepthState.PASS, "CONSISTENT_PROTRUSION", fraction, prominent)
    if float(np.percentile(np.abs(heights), 90)) <= cfg.depth_flat_mm:
        return DepthEvidence(DepthState.FLAT, "CONSISTENT_PLANE", fraction, prominent)
    return DepthEvidence(DepthState.UNKNOWN, "AMBIGUOUS_GEOMETRY", fraction, prominent)


def track_decision(class_id: int, states: Iterable[DepthState], hits: int,
                   possible_stack: bool = False, round_table: bool = False,
                   cfg: Thresholds = DEFAULTS):
    """Depth may only veto a stable YOLO track; it cannot create one."""
    if hits < (cfg.min_track_hits_round if round_table else cfg.min_track_hits_square):
        return False, "YOLO_UNSTABLE"
    states = list(states)
    if class_id in WATER_CLASS_IDS or possible_stack:
        return True, "DEPTH_NONVETO"
    if states.count(DepthState.FLAT) >= cfg.flat_min_frames and DepthState.PASS not in states:
        return False, "REPEATED_FLAT"
    return True, "YOLO_STABLE"


def water_capacity_decision(scores: dict[int, float], hits: int, ocr_ml: int | None = None,
                            size_vote: int | None = None, cfg: Thresholds = DEFAULTS):
    """Return class ID 10/11 or None. OCR/size only help stable YOLO evidence."""
    if hits < cfg.water_min_hits:
        return None
    a, b = float(scores.get(10, 0)), float(scores.get(11, 0))
    if not all(isfinite(x) and x >= 0 for x in (a, b)) or a + b <= 0:
        return None
    if ocr_ml not in (None, 350, 550) or size_vote not in (None, 350, 550):
        raise ValueError("Capacity evidence must be explicit 350/550 or None")
    if ocr_ml == 350:
        a += 0.5
    elif ocr_ml == 550:
        b += 0.5
    if size_vote == 350:
        a += 0.2
    elif size_vote == 550:
        b += 0.2
    margin = abs(a-b) / (a+b)
    if margin < cfg.water_min_margin:
        return None
    winner = 10 if a > b else 11
    if ocr_ml is not None and ((winner == 10 and ocr_ml != 350) or (winner == 11 and ocr_ml != 550)):
        return None
    return winner
