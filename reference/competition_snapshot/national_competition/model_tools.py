"""Inspect, export, and compare a candidate without touching provincial models."""
import argparse
import hashlib
import json
from pathlib import Path

from .config import NAMES, validate_model_names


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pt(path):
    from ultralytics import YOLO
    import ultralytics

    model = YOLO(str(path))
    if model.task != "detect":
        raise ValueError(f"Expected detect task, got {model.task}")
    validate_model_names(model.names)
    return {"file": str(path.resolve()), "sha256": sha256(path), "task": model.task,
            "names": list(NAMES), "ultralytics": ultralytics.__version__}


def inspect_onnx(path):
    import onnx

    model = onnx.load(str(path))
    shape = lambda value: [dim.dim_value if dim.dim_value else dim.dim_param
                           for dim in value.type.tensor_type.shape.dim]
    inputs = [{"name": x.name, "shape": shape(x)} for x in model.graph.input]
    outputs = [{"name": x.name, "shape": shape(x)} for x in model.graph.output]
    runtime_validator = None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        runtime_inputs = [{"name": x.name, "shape": list(x.shape), "type": x.type}
                          for x in session.get_inputs()]
        runtime_outputs = [{"name": x.name, "shape": list(x.shape), "type": x.type}
                           for x in session.get_outputs()]
        runtime_validator = "onnxruntime"
    except (ImportError, OSError):
        import cv2
        net = cv2.dnn.readNetFromONNX(str(path))
        if not net.getLayerNames():
            raise ValueError("OpenCV DNN loaded no ONNX layers")
        runtime_inputs, runtime_outputs = inputs, outputs
        runtime_validator = "opencv-dnn"
    if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 640, 640]:
        raise ValueError(f"Expected fixed 1x3x640x640 input: {inputs}")
    if len(outputs) != 1:
        raise ValueError(f"Expected one raw YOLO output: {outputs}")
    metadata = {p.key: p.value for p in model.metadata_props}
    if "names" in metadata:
        import ast
        validate_model_names(ast.literal_eval(metadata["names"]))
    return {"file": str(path.resolve()), "sha256": sha256(path), "opset":
            [x.version for x in model.opset_import], "inputs": inputs, "outputs": outputs,
            "runtime_inputs": runtime_inputs, "runtime_outputs": runtime_outputs,
            "runtime_validator": runtime_validator,
            "metadata_names_present": "names" in metadata}


def export_pt(path, outdir, name):
    if not name.startswith("national_") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in name):
        raise ValueError("Use a distinct national_<version> name")
    info = inspect_pt(path)
    output = Path(outdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_pt = output / f"{name}.pt"
    if target_pt.exists() or (output / f"{name}.onnx").exists():
        raise FileExistsError("Version already exists; never overwrite a model")
    import shutil
    shutil.copy2(path, target_pt)
    if sha256(target_pt) != info["sha256"]:
        raise RuntimeError("Copied PT hash mismatch")
    from ultralytics import YOLO
    exported = Path(YOLO(str(target_pt)).export(format="onnx", imgsz=640, opset=12,
                                                   dynamic=False, nms=False, simplify=False,
                                                   batch=1, device="cpu"))
    target_onnx = output / f"{name}.onnx"
    if exported.resolve() != target_onnx:
        exported.rename(target_onnx)
    onnx_info = inspect_onnx(target_onnx)
    manifest = {"version": name, "pt": info, "onnx": onnx_info,
                "export": {"imgsz": 640, "batch": 1, "opset": 12, "dynamic": False,
                           "nms": False, "simplify": False, "device": "cpu"},
                "postprocess": "external; compare exact preprocessing and NMS with Windows"}
    (output / f"{name}.export.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def finalize_export(pt, onnx, outdir, name):
    output = Path(outdir).resolve()
    pt_info = inspect_pt(pt)
    onnx_info = inspect_onnx(onnx)
    manifest = {"version": name, "pt": pt_info, "onnx": onnx_info,
                "export": {"imgsz": 640, "batch": 1, "opset": 12, "dynamic": False,
                           "nms": False, "simplify": False, "device": "cpu"},
                "postprocess": "external; class-agnostic NMS and TABLE filtering"}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.export.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def compare_raw(reference, candidate):
    import numpy as np
    a = np.load(reference)
    b = np.load(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Output shape mismatch: {a.shape} vs {b.shape}")
    delta = np.abs(a.astype("float32") - b.astype("float32"))
    return {"shape": list(a.shape), "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()), "p99_abs": float(np.percentile(delta, 99))}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("pt", type=Path)
    export = commands.add_parser("export")
    export.add_argument("pt", type=Path)
    export.add_argument("outdir", type=Path)
    export.add_argument("version")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("pt", type=Path)
    finalize.add_argument("onnx", type=Path)
    finalize.add_argument("outdir", type=Path)
    finalize.add_argument("version")
    compare = commands.add_parser("compare-raw")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = (inspect_pt(args.pt) if args.command == "inspect" else
              export_pt(args.pt, args.outdir, args.version) if args.command == "export" else
              finalize_export(args.pt, args.onnx, args.outdir, args.version)
              if args.command == "finalize" else
              compare_raw(args.reference, args.candidate))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
