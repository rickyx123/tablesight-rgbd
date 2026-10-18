"""One canonical class order for Windows, ONNX, OM, and result generation."""
from dataclasses import dataclass

NAMES = (
    "眼镜", "胶带", "遥控器", "卷尺", "调味料", "桶装方便面", "面包", "薯片",
    "罐装饮料", "袋装饮料", "350ml瓶装饮用水", "550ml瓶装饮用水",
    "石榴", "柿子", "青枣/冬枣", "柚子", "桌子",
)
GOAL_IDS = (
    "CA001", "CA002", "CA003", "CA004", "CB001", "CB002", "CB003", "CB004",
    "CC001", "CC002", "CC003", "CC004", "CD001", "CD002", "CD003", "CD004", None,
)
TABLE_CLASS_ID = 16
WATER_CLASS_IDS = (10, 11)
MODEL_VERSION = "national_A_20260927"


@dataclass(frozen=True)
class Thresholds:
    object_conf: float = 0.35
    table_conf: float = 0.25
    nms_iou: float = 0.55
    competition_iou: float = 0.85
    competition_area_ratio: float = 0.75
    table_margin_fraction: float = 0.02
    min_track_hits_square: int = 3
    min_track_hits_round: int = 3
    flat_min_frames: int = 3
    depth_min_valid_fraction: float = 0.55
    depth_min_ring_pixels: int = 60
    depth_pass_mm: float = 18.0
    depth_flat_mm: float = 6.0
    water_min_hits: int = 3
    water_min_margin: float = 0.35


DEFAULTS = Thresholds()
CLASS_CONF_OVERRIDES = {0: 0.60, 1: 0.45, 2: 0.50, 4: 0.45, 6: 0.70, 8: 0.70, 10: 0.45, 11: 0.45}
# Retain modest per-frame scores for voting; reject weak final identity tracks.
CLASS_TRACK_CONF_OVERRIDES = {2: 0.55, 4: 0.50}


def validate_model_names(names):
    if isinstance(names, dict):
        ordered = [names[k] if k in names else names[str(k)] for k in range(len(names))]
    else:
        ordered = list(names)
    if ordered != list(NAMES):
        mismatches = [(i, expected, ordered[i] if i < len(ordered) else "<missing>")
                      for i, expected in enumerate(NAMES)
                      if i >= len(ordered) or ordered[i] != expected]
        raise ValueError(f"National class order mismatch: {mismatches}; count={len(ordered)}")
    return True
