"""Machine-readable inference contract shared with the Windows implementation."""
import json
from dataclasses import asdict

from .config import (CLASS_CONF_OVERRIDES, DEFAULTS, GOAL_IDS, MODEL_VERSION,
                     NAMES, TABLE_CLASS_ID)


def contract():
    return {"model_version": MODEL_VERSION,
            "classes": [{"index": i, "name": name, "goal_id": GOAL_IDS[i]}
                        for i, name in enumerate(NAMES)],
            "input": {"shape": [1, 3, 640, 640], "dtype": "float32", "color": "RGB",
                      "scale": "pixel/255", "letterbox_fill_bgr": [114, 114, 114],
                      "letterbox_interpolation": "cv2.INTER_LINEAR"},
            "output": {"embedded_nms": False, "decoder": "verify against candidate PT/ONNX"},
            "nms": {"candidate_iou": DEFAULTS.nms_iou,
                    "cross_class_competing_iou": DEFAULTS.competition_iou,
                    "mode": "candidate behavior pending PT/Windows comparison"},
            "table_class_id": TABLE_CLASS_ID,
            "thresholds": asdict(DEFAULTS),
            "class_conf_overrides": CLASS_CONF_OVERRIDES}


if __name__ == "__main__":
    print(json.dumps(contract(), ensure_ascii=False, indent=2))
