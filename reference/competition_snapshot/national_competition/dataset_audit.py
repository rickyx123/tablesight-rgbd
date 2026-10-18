"""Read-only report on current LabelMe data; never modifies source files."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .config import NAMES


def audit(root):
    root = Path(root)
    images = list(root.rglob("*.jpg"))
    annotations = list(root.rglob("*.json"))
    labels = Counter()
    errors = []
    paired = 0
    mirrored = 0
    for path in annotations:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            labels.update(shape.get("label", "") for shape in item.get("shapes", []))
            if path.with_suffix(".jpg").exists():
                paired += 1
            elif path.parent.name == "labels" and (root / "images" / (path.stem + ".jpg")).exists():
                mirrored += 1
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
    return {"root": str(root.resolve()), "jpg": len(images), "json": len(annotations),
            "paired_same_directory": paired, "paired_labels_to_images": mirrored,
            "unpaired_json": len(annotations) - paired - mirrored,
            "unrecognized_labels":
            {k: v for k, v in labels.items() if k not in NAMES},
            "label_counts": dict(labels), "errors": errors,
            "total_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))
