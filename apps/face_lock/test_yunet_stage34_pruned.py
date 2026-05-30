import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OLD_MODEL = APP_DIR / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
DEFAULT_PRUNED_MODEL = APP_DIR / "models" / "yunet" / "face_detection_yunet_stage2_dropped.onnx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGE_NAME_HINTS = {
    "stage3": ("stage3", "stage 3", "stride16", "scale16", "out3"),
    "stage4": ("stage4", "stage 4", "stride32", "scale32", "out4"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare stage-3 and stage-4 outputs between the old YuNet model and the pruned YuNet model."
    )
    parser.add_argument("--old-model", type=Path, default=DEFAULT_OLD_MODEL, help="Path to the original YuNet ONNX model.")
    parser.add_argument("--pruned-model", type=Path, default=DEFAULT_PRUNED_MODEL, help="Path to the pruned YuNet ONNX model.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Optional directory of test images.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of deterministic inputs to compare.")
    parser.add_argument("--input-height", type=int, default=640, help="Fallback input height when the model shape is dynamic.")
    parser.add_argument("--input-width", type=int, default=640, help="Fallback input width when the model shape is dynamic.")
    parser.add_argument("--stage3-output", type=str, default=None, help="Explicit stage-3 output name to compare.")
    parser.add_argument("--stage4-output", type=str, default=None, help="Explicit stage-4 output name to compare.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for np.testing.assert_allclose.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for np.testing.assert_allclose.")
    return parser.parse_args()


def build_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def resolve_input_shape(session, fallback_height: int, fallback_width: int):
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4:
        raise RuntimeError(f"Expected a 4D input tensor, but got shape metadata: {shape}")

    height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else fallback_height
    width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else fallback_width
    return int(height), int(width)


def load_image(path: Path, height: int, width: int):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode image: {path}")
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(frame)


def make_synthetic_inputs(num_samples: int, height: int, width: int):
    samples = []

    zeros = np.zeros((1, 3, height, width), dtype=np.float32)
    samples.append(zeros)

    if num_samples > 1:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        ramp = np.stack(
            [
                np.broadcast_to(x, (height, width)),
                np.broadcast_to(y, (height, width)),
                np.broadcast_to(1.0 - x, (height, width)),
            ],
            axis=0,
        )[None, ...]
        samples.append(np.ascontiguousarray(ramp))

    rng = np.random.default_rng(20260530)
    while len(samples) < num_samples:
        samples.append(np.ascontiguousarray(rng.random((1, 3, height, width), dtype=np.float32)))

    return samples[:num_samples]


def collect_inputs(args, height, width):
    if args.input_dir is None:
        return make_synthetic_inputs(max(1, args.num_samples), height, width)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported image files found in {args.input_dir}")

    inputs = [load_image(path, height, width) for path in paths[: max(1, args.num_samples)]]
    return inputs


def infer_output_names(session, requested_name, stage_label):
    available = [output.name for output in session.get_outputs()]
    if requested_name is not None:
        if requested_name not in available:
            raise RuntimeError(
                f"Requested {stage_label} output '{requested_name}' was not found. Available outputs: {available}"
            )
        return requested_name

    if len(available) == 2:
        return available[0] if stage_label == "stage3" else available[1]

    lowered = {name: name.lower() for name in available}
    candidates = [name for name, lower in lowered.items() if any(token in lower for token in STAGE_NAME_HINTS[stage_label])]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not automatically resolve {stage_label} output for outputs {available}. "
        f"Pass --{stage_label}-output explicitly."
    )


def compare_outputs(old_session, pruned_session, inputs, old_stage_names, pruned_stage_names, rtol, atol):
    old_input_name = old_session.get_inputs()[0].name
    pruned_input_name = pruned_session.get_inputs()[0].name

    stats = {
        "stage3": {"max_abs": 0.0, "sum_abs": 0.0, "count": 0},
        "stage4": {"max_abs": 0.0, "sum_abs": 0.0, "count": 0},
    }

    for sample_index, sample in enumerate(inputs):
        old_outputs = {
            "stage3": old_session.run([old_stage_names["stage3"]], {old_input_name: sample})[0],
            "stage4": old_session.run([old_stage_names["stage4"]], {old_input_name: sample})[0],
        }
        pruned_outputs = {
            "stage3": pruned_session.run([pruned_stage_names["stage3"]], {pruned_input_name: sample})[0],
            "stage4": pruned_session.run([pruned_stage_names["stage4"]], {pruned_input_name: sample})[0],
        }

        for stage_label in ("stage3", "stage4"):
            old_value = old_outputs[stage_label]
            pruned_value = pruned_outputs[stage_label]
            if old_value.shape != pruned_value.shape:
                raise RuntimeError(
                    f"{stage_label} shape mismatch on sample {sample_index}: "
                    f"old={old_value.shape}, pruned={pruned_value.shape}"
                )

            diff = np.abs(old_value - pruned_value)
            stats[stage_label]["max_abs"] = max(stats[stage_label]["max_abs"], float(diff.max()))
            stats[stage_label]["sum_abs"] += float(diff.sum())
            stats[stage_label]["count"] += diff.size

            np.testing.assert_allclose(
                old_value,
                pruned_value,
                rtol=rtol,
                atol=atol,
                err_msg=f"{stage_label} output mismatch on sample {sample_index}",
            )

    return stats


def print_summary(stats, old_stage_names, pruned_stage_names):
    print("\nYuNet Stage 3/4 Comparison")
    print("=" * 78)
    print(f"{'Stage':<10} | {'Old Output':<30} | {'Pruned Output':<30} | {'Max Abs':>10} | {'Mean Abs':>10}")
    print("-" * 78)
    for stage_label in ("stage3", "stage4"):
        count = max(1, stats[stage_label]["count"])
        mean_abs = stats[stage_label]["sum_abs"] / count
        print(
            f"{stage_label:<10} | "
            f"{old_stage_names[stage_label]:<30} | "
            f"{pruned_stage_names[stage_label]:<30} | "
            f"{stats[stage_label]['max_abs']:>10.6f} | "
            f"{mean_abs:>10.6f}"
        )
        print(f"  {stage_label}: PASS")
    print("=" * 78)


def main():
    args = parse_args()

    old_session = build_session(args.old_model)
    pruned_session = build_session(args.pruned_model)

    old_height, old_width = resolve_input_shape(old_session, args.input_height, args.input_width)
    pruned_height, pruned_width = resolve_input_shape(pruned_session, args.input_height, args.input_width)
    if (old_height, old_width) != (pruned_height, pruned_width):
        raise RuntimeError(
            f"Input shape mismatch: old=({old_height}, {old_width}), pruned=({pruned_height}, {pruned_width})"
        )

    old_inputs = collect_inputs(args, old_height, old_width)

    old_stage_names = {
        "stage3": infer_output_names(old_session, args.stage3_output, "stage3"),
        "stage4": infer_output_names(old_session, args.stage4_output, "stage4"),
    }
    pruned_stage_names = {
        "stage3": infer_output_names(pruned_session, args.stage3_output, "stage3"),
        "stage4": infer_output_names(pruned_session, args.stage4_output, "stage4"),
    }

    if old_stage_names["stage3"] == old_stage_names["stage4"]:
        raise RuntimeError("Stage 3 and stage 4 resolved to the same output in the old model.")
    if pruned_stage_names["stage3"] == pruned_stage_names["stage4"]:
        raise RuntimeError("Stage 3 and stage 4 resolved to the same output in the pruned model.")

    stats = compare_outputs(
        old_session,
        pruned_session,
        old_inputs,
        old_stage_names,
        pruned_stage_names,
        rtol=args.rtol,
        atol=args.atol,
    )
    print_summary(stats, old_stage_names, pruned_stage_names)
    print("Stage 3 and stage 4 outputs match within tolerance.")


if __name__ == "__main__":
    sys.exit(main())
