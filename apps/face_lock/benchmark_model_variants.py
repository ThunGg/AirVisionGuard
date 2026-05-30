import argparse
import ctypes
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
import psutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from detector import MTCNNDetector, YuNetDetector
from utils import align_face, preprocess_face


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

DEFAULT_PTH_PATH = REPO_ROOT / "pretrained_models" / "ckpt_epoch_40.pth.tar"
DEFAULT_ONNX_PATH = REPO_ROOT / "pretrained_models" / "model.onnx"
DEFAULT_QUANT_PATH = APP_DIR / "models" / "model_quant.onnx"
DEFAULT_DETECTOR_MODEL = APP_DIR / "models" / "yunet" / "yunet_n_640_640.onnx"
OLD_DETECTOR_MODEL = APP_DIR / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
STAGE2_DROPPED_DETECTOR_MODEL = APP_DIR / "models" / "yunet" / "face_detection_yunet_stage2_dropped.onnx"
YUNET_VARIANT_MODELS = {
    "default": DEFAULT_DETECTOR_MODEL,
    "old": OLD_DETECTOR_MODEL,
    "old-stage2-dropped": STAGE2_DROPPED_DETECTOR_MODEL,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MB = 1024 * 1024

KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
PSAPI = ctypes.WinDLL("psapi", use_last_error=True)


@dataclass(frozen=True)
class StageConfig:
    label: str
    target_fps: float
    compare_authorized: bool
    enable_skip_cache: bool
    collect_registration: bool = False


@dataclass
class VariantRuntime:
    label: str
    recognizer: object
    runtime: str
    power_device: str
    disk_size_mb: float
    ram_added_mb: float
    path: str


@dataclass
class FrameBuckets:
    face_frames: list
    no_face_frames: list
    synthetic_no_face_frames: bool = False


STAGES = [
    StageConfig("Registering", target_fps=8.0, compare_authorized=False, enable_skip_cache=False, collect_registration=True),
    StageConfig("Detecting", target_fps=8.0, compare_authorized=True, enable_skip_cache=False),
    StageConfig("Authorized Cached", target_fps=2.0, compare_authorized=True, enable_skip_cache=True),
    StageConfig("Locked", target_fps=3.0, compare_authorized=True, enable_skip_cache=False),
]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Face Lock end-to-end stage performance for original, ONNX, and quantized ONNX recognition variants."
    )
    parser.add_argument("--pth-path", type=Path, default=DEFAULT_PTH_PATH, help="Path to the original PyTorch checkpoint.")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH, help="Path to the exported ONNX model.")
    parser.add_argument("--quant-path", type=Path, default=DEFAULT_QUANT_PATH, help="Path to the quantized ONNX model.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Primary device for original and ONNX variants.")
    parser.add_argument("--quant-device", choices=["same", "cpu", "cuda"], default="same", help="Inference device for the quantized ONNX variant.")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for CUDA execution.")
    parser.add_argument("--backbone", type=str, default="auto", help="Backbone name for the original checkpoint. Use 'auto' to read the checkpoint arch field.")
    parser.add_argument("--feature-dim", type=int, default=512, help="Embedding size for the original model.")
    parser.add_argument("--input-size", type=int, default=112, help="Input height/width for face preprocessing.")
    parser.add_argument("--feat-bn", action=argparse.BooleanOptionalAction, default=True, help="Whether the original framework model uses feature batch norm.")
    parser.add_argument("--auto-quantize", action=argparse.BooleanOptionalAction, default=True, help="Create the quantized ONNX model automatically when it is missing.")
    parser.add_argument("--detector", choices=["yunet", "mtcnn"], default="yunet", help="Face detector backend to use during the practical benchmark.")
    parser.add_argument("--detector-model", type=Path, default=None, help="Explicit path to the YuNet detector ONNX model. Overrides --yunet-variant when provided.")
    parser.add_argument(
        "--yunet-variant",
        choices=list(YUNET_VARIANT_MODELS.keys()),
        default="default",
        help="Preset YuNet detector variant to use when --detector-model is not set.",
    )
    parser.add_argument("--source", choices=["webcam", "images"], default="webcam", help="Frame source for the benchmark.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Directory of benchmark frames when --source images is used.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index when --source webcam is used.")
    parser.add_argument("--frame-count", type=int, default=24, help="Number of shared frames to collect and replay per stage.")
    parser.add_argument("--camera-warmup", type=int, default=10, help="Number of warmup webcam reads before collecting benchmark frames.")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested webcam width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested webcam height.")
    parser.add_argument("--camera-fps", type=int, default=15, help="Requested webcam FPS.")
    parser.add_argument("--max-capture-frames", type=int, default=240, help="Maximum raw frames to inspect while gathering benchmark samples.")
    parser.add_argument("--registration-steps", type=int, default=3, help="Number of embeddings to collect during the registering stage.")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="Cosine similarity threshold used by Face Lock.")
    parser.add_argument("--detect-every-n", type=int, default=3, help="Run face detection every Nth frame, matching the app loop.")
    parser.add_argument("--skip-after-auth", type=int, default=15, help="Skip recognition for N frames after authorization.")
    parser.add_argument("--simulate-throttle", action=argparse.BooleanOptionalAction, default=True, help="Sleep to match the app's stage FPS targets.")
    parser.add_argument("--limit-opencv-threads", type=int, default=2, help="OpenCV thread limit to match the app.")
    return parser.parse_args()


def get_rss_mb():
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p
    PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
    PSAPI.GetProcessMemoryInfo.restype = ctypes.c_int
    process = KERNEL32.GetCurrentProcess()
    ok = PSAPI.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return counters.WorkingSetSize / MB


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_file_size_mb(path: Path):
    total = path.stat().st_size
    external_data_path = Path(str(path) + ".data")
    if external_data_path.exists():
        total += external_data_path.stat().st_size
    return total / MB


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device_arg


def resolve_quant_device(device_arg: str, default_device: str):
    if device_arg == "same":
        return default_device
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the quantized model but is not available.")
    return device_arg


def normalize_embedding(output):
    output = output / np.linalg.norm(output, axis=1, keepdims=True)
    return output[0]


def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2))


def load_checkpoint_state(model, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    normalized_state = {}
    for key, value in state_dict.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        if clean_key.startswith("basemodel.") or clean_key.startswith("bn1d."):
            normalized_state[clean_key] = value
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    return checkpoint, missing, unexpected


def infer_backbone(checkpoint_path: Path, fallback_backbone: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint.get("arch"):
        return checkpoint["arch"]
    if fallback_backbone == "auto":
        raise RuntimeError("Could not infer the backbone from the checkpoint. Please pass --backbone explicitly.")
    return fallback_backbone


def load_backbone_factory(backbone_name: str):
    if backbone_name not in {"mobilefacenet", "mobilefacenet_large"}:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}' for this app benchmark. "
            "Use the Face Lock MobileFaceNet checkpoint or extend this script with the required backbone factory."
        )

    backbone_module_path = REPO_ROOT / "models" / "backbones" / "mobilefacenet.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("face_lock_mobilefacenet", backbone_module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, backbone_name)


class OriginalInferenceModel(nn.Module):
    def __init__(self, backbone_name: str, feature_dim: int, input_size: int, feat_bn: bool):
        super().__init__()
        backbone_factory = load_backbone_factory(backbone_name)
        self.basemodel = backbone_factory(feature_dim=feature_dim, spatial_size=input_size)
        self.use_feat_bn = feat_bn
        if feat_bn:
            self.bn1d = nn.BatchNorm1d(feature_dim, affine=False, eps=2e-5, momentum=0.9)

    def forward(self, x, extract_mode=True):
        feature = self.basemodel(x)
        if self.use_feat_bn:
            feature = self.bn1d(feature)
        return feature


class OriginalRecognizer:
    def __init__(self, args, device: str):
        backbone = infer_backbone(args.pth_path, args.backbone)
        self.model = OriginalInferenceModel(
            backbone_name=backbone,
            feature_dim=args.feature_dim,
            input_size=args.input_size,
            feat_bn=args.feat_bn,
        )
        checkpoint, missing, unexpected = load_checkpoint_state(self.model, args.pth_path)
        if missing:
            print(f"[original] Missing checkpoint keys: {missing}")
        if unexpected:
            print(f"[original] Unexpected checkpoint keys: {unexpected}")
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.runtime = f"PyTorch on {device} ({checkpoint.get('arch', backbone) if isinstance(checkpoint, dict) else backbone})"
        self.power_device = device

    def get_embedding(self, img_tensor):
        tensor = torch.from_numpy(img_tensor).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor, extract_mode=True).detach().cpu().numpy()
        return normalize_embedding(output)

    def compute_similarity(self, emb1, emb2):
        return cosine_similarity(emb1, emb2)


class OrtRecognizer:
    def __init__(self, model_path: Path, requested_device: str, gpu_index: int):
        self.session, provider = build_session(model_path, requested_device, gpu_index)
        self.input_name = self.session.get_inputs()[0].name
        self.runtime = provider
        self.power_device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"

    def get_embedding(self, img_tensor):
        output = self.session.run(None, {self.input_name: img_tensor})[0]
        return normalize_embedding(output)

    def compute_similarity(self, emb1, emb2):
        return cosine_similarity(emb1, emb2)


def build_session(path: Path, requested_device: str, gpu_index: int):
    providers = ["CPUExecutionProvider"]
    provider_used = "CPUExecutionProvider"
    if requested_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = [("CUDAExecutionProvider", {"device_id": gpu_index}), "CPUExecutionProvider"]
        provider_used = "CUDAExecutionProvider"
    session = ort.InferenceSession(str(path), providers=providers)
    actual_provider = session.get_providers()[0] if session.get_providers() else provider_used
    return session, actual_provider


def ensure_quantized_model(onnx_path: Path, quant_path: Path):
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    packed_model_path = quant_path.with_name(f"{quant_path.stem}_source.onnx")
    packed_model_created = False
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx

    print(f"Creating quantized ONNX model at {quant_path} ...")
    if Path(str(onnx_path) + ".data").exists():
        model = onnx.load_model(str(onnx_path), load_external_data=True)
        onnx.save_model(model, str(packed_model_path), save_as_external_data=False)
        quant_input_path = packed_model_path
        packed_model_created = True
    else:
        quant_input_path = onnx_path

    local_temp = APP_DIR / ".quant_tmp"
    local_temp.mkdir(parents=True, exist_ok=True)
    os_tmp = os.environ.get("TMP")
    os_temp = os.environ.get("TEMP")
    try:
        os.environ["TMP"] = str(local_temp)
        os.environ["TEMP"] = str(local_temp)
        tempfile.tempdir = str(local_temp)
        quantize_dynamic(str(quant_input_path), str(quant_path), weight_type=QuantType.QUInt8)
    except Exception as exc:
        raise RuntimeError(
            "Automatic quantized ONNX generation failed. "
            "Provide --quant-path with an existing quantized model, or run the script in an environment "
            "where ONNX Runtime quantization can write temporary files."
        ) from exc
    finally:
        tempfile.tempdir = None
        if os_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = os_tmp
        if os_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = os_temp

    if packed_model_created:
        try:
            packed_model_path.unlink()
        except OSError:
            pass

    print("Quantized ONNX model created successfully.")


def measure_ram_added(load_func):
    clear_memory()
    before = get_rss_mb()
    instance = load_func()
    clear_memory()
    after = get_rss_mb()
    return instance, max(0.0, after - before)


def measure_stage(stage_callable, gpu_index: int):
    p = psutil.Process(os.getpid())
    p.cpu_percent(interval=None)
    start = time.perf_counter()

    payload = stage_callable()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(torch.device(f"cuda:{gpu_index}"))
        except:
            pass

    elapsed_s = time.perf_counter() - start
    cpu_usage = p.cpu_percent(interval=None)

    return payload, elapsed_s, cpu_usage, "psutil"


def create_detector(args, detector_model: Path | None = None):
    if args.detector == "mtcnn":
        return MTCNNDetector()
    model_path = detector_model if detector_model is not None else args.detector_model
    if model_path is None:
        raise RuntimeError("YuNet detector model path was not resolved.")
    return YuNetDetector(model_path=str(model_path))


def resolve_detector_model(args):
    if args.detector != "yunet":
        return None
    if args.detector_model is not None:
        return args.detector_model
    return YUNET_VARIANT_MODELS[args.yunet_variant]


def resolve_bucket_detector_model(args):
    if args.detector != "yunet":
        return None
    # Keep benchmark frame bucketing stable across YuNet variants so stage
    # replay uses the same face/no-face mix for old and pruned detector runs.
    return OLD_DETECTOR_MODEL


def load_webcam_frames(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the benchmark camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        for _ in range(max(0, args.camera_warmup)):
            cap.read()

        frames = []
        while len(frames) < args.max_capture_frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a webcam frame during benchmark collection.")
            frames.append(frame.copy())
        return frames
    finally:
        cap.release()


def load_image_frames(args):
    if args.image_dir is None:
        raise RuntimeError("Please provide --image-dir when using --source images.")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No supported image files found in {args.image_dir}")

    frames = []
    for path in paths[: args.max_capture_frames]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"Could not decode any images from {args.image_dir}")
    return frames


def load_benchmark_frames(args):
    if args.source == "webcam":
        return load_webcam_frames(args)
    return load_image_frames(args)


def make_synthetic_no_face_frames(face_frames, frame_count):
    if not face_frames:
        raise RuntimeError("Cannot synthesize no-face frames because no benchmark frames were collected.")
    height, width = face_frames[0].shape[:2]
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    return [blank.copy() for _ in range(frame_count)]


def bucket_frames_by_detection(frames, detector, args):
    face_frames = []
    no_face_frames = []

    for frame in frames:
        boxes, _ = detector.detect(frame)
        if boxes.shape[0] > 0:
            if len(face_frames) < args.frame_count:
                face_frames.append(frame.copy())
        else:
            if len(no_face_frames) < args.frame_count:
                no_face_frames.append(frame.copy())

        if len(face_frames) >= args.frame_count and len(no_face_frames) >= args.frame_count:
            break

    if len(face_frames) < args.registration_steps:
        raise RuntimeError(
            f"Only found {len(face_frames)} usable face frames during collection, but registration needs "
            f"{args.registration_steps}. Try facing the camera more clearly, increasing --max-capture-frames, "
            f"or use --source images with face-containing samples."
        )

    synthetic_no_face_frames = False
    if not no_face_frames:
        no_face_frames = make_synthetic_no_face_frames(face_frames, args.frame_count)
        synthetic_no_face_frames = True

    while len(face_frames) < args.frame_count:
        face_frames.append(face_frames[len(face_frames) % max(1, len(face_frames))].copy())

    while len(no_face_frames) < args.frame_count:
        no_face_frames.append(no_face_frames[len(no_face_frames) % max(1, len(no_face_frames))].copy())

    return FrameBuckets(
        face_frames=face_frames[: args.frame_count],
        no_face_frames=no_face_frames[: args.frame_count],
        synthetic_no_face_frames=synthetic_no_face_frames,
    )


def select_largest_face(boxes, landmarks):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    max_idx = int(areas.argmax())
    return boxes[max_idx], landmarks[max_idx]


def run_stage(stage: StageConfig, frames, detector, recognizer, auth_embeddings, args):
    processing_ms = []
    loop_ms = []
    face_detections = 0
    recognition_calls = 0
    authorized_hits = 0
    working_auth = [emb.copy() for emb in auth_embeddings]
    skip_recognition_count = 0
    cached_boxes = None
    cached_landmarks = None

    frame_total = len(frames)
    for frame_idx in range(frame_total):
        frame = frames[frame_idx % len(frames)]
        loop_start = time.perf_counter()

        if frame_idx % args.detect_every_n == 0 or cached_boxes is None:
            boxes, landmarks = detector.detect(frame)
            cached_boxes, cached_landmarks = boxes, landmarks
        else:
            boxes, landmarks = cached_boxes, cached_landmarks

        if boxes.shape[0] > 0:
            face_detections += 1
            box, landmark = select_largest_face(boxes, landmarks)

            if (
                stage.enable_skip_cache
                and skip_recognition_count > 0
                and len(working_auth) >= args.registration_steps
            ):
                skip_recognition_count -= 1
                authorized_hits += 1
            else:
                aligned_face = align_face(frame, box, landmark, image_size=args.input_size)
                img_tensor = preprocess_face(aligned_face)
                embedding = recognizer.get_embedding(img_tensor)
                recognition_calls += 1

                if stage.collect_registration and len(working_auth) < args.registration_steps:
                    working_auth.append(embedding)

                if stage.compare_authorized and len(working_auth) >= args.registration_steps:
                    similarities = [recognizer.compute_similarity(embedding, auth_emb) for auth_emb in working_auth]
                    if max(similarities) > args.similarity_threshold:
                        authorized_hits += 1
                        if stage.enable_skip_cache:
                            skip_recognition_count = args.skip_after_auth

        processing_elapsed = time.perf_counter() - loop_start
        processing_ms.append(processing_elapsed * 1000.0)

        if args.simulate_throttle and stage.target_fps > 0:
            target_frame_s = 1.0 / stage.target_fps
            remaining_s = max(0.0, target_frame_s - processing_elapsed)
            if remaining_s > 0:
                time.sleep(remaining_s)

        loop_ms.append((time.perf_counter() - loop_start) * 1000.0)

    if stage.collect_registration and len(working_auth) < args.registration_steps:
        raise RuntimeError(
            f"Registration stage only found {len(working_auth)} usable face embeddings. "
            f"Need {args.registration_steps}. Try a longer run or clearer face frames."
        )

    return {
        "processing_ms_mean": float(np.mean(processing_ms)) if processing_ms else None,
        "latency_ms_mean": float(np.mean(loop_ms)) if loop_ms else None,
        "frames_processed": frame_total,
        "detections": face_detections,
        "recognition_calls": recognition_calls,
        "authorized_hits": authorized_hits,
        "auth_embeddings": working_auth,
    }


def build_variant_runtimes(args, device: str, quant_device: str):
    original_recognizer, original_ram = measure_ram_added(lambda: OriginalRecognizer(args, device))
    onnx_recognizer, onnx_ram = measure_ram_added(lambda: OrtRecognizer(args.onnx_path, device, args.gpu_index))
    quant_recognizer, quant_ram = measure_ram_added(lambda: OrtRecognizer(args.quant_path, quant_device, args.gpu_index))

    return {
        "original": VariantRuntime(
            label="Original",
            recognizer=original_recognizer,
            runtime=original_recognizer.runtime,
            power_device=original_recognizer.power_device,
            disk_size_mb=get_file_size_mb(args.pth_path),
            ram_added_mb=original_ram,
            path=str(args.pth_path),
        ),
        "onnx": VariantRuntime(
            label="ONNX",
            recognizer=onnx_recognizer,
            runtime=onnx_recognizer.runtime,
            power_device=onnx_recognizer.power_device,
            disk_size_mb=get_file_size_mb(args.onnx_path),
            ram_added_mb=onnx_ram,
            path=str(args.onnx_path),
        ),
        "quant": VariantRuntime(
            label="Quant ONNX",
            recognizer=quant_recognizer,
            runtime=quant_recognizer.runtime,
            power_device=quant_recognizer.power_device,
            disk_size_mb=get_file_size_mb(args.quant_path),
            ram_added_mb=quant_ram,
            path=str(args.quant_path),
        ),
    }


def benchmark_variant_stages(variant: VariantRuntime, frames, detector, args):
    stage_results = {}
    auth_embeddings = []

    for stage in STAGES:
        stage_frames = frames.face_frames if stage.label != "Locked" else frames.no_face_frames

        def run_callable():
            return run_stage(stage, stage_frames, detector, variant.recognizer, auth_embeddings, args)

        payload, elapsed_s, cpu_usage, measurement_source = measure_stage(
            run_callable,
            gpu_index=args.gpu_index,
        )

        auth_embeddings = payload["auth_embeddings"]
        stage_results[stage.label] = {
            "processing_ms": payload["processing_ms_mean"],
            "latency_ms": payload["latency_ms_mean"],
            "throughput_fps": (payload["frames_processed"] / elapsed_s) if elapsed_s > 0 else None,
            "cpu_usage": cpu_usage,
            "measurement_source": measurement_source,
            "detections": payload["detections"],
            "recognition_calls": payload["recognition_calls"],
            "authorized_hits": payload["authorized_hits"],
            "frames_processed": payload["frames_processed"],
        }

    return stage_results


def format_float(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def print_load_table(variants):
    print("\nLoad Metrics")
    print("=" * 72)
    print(f"{'Metric':<20} | {'Original':>14} | {'ONNX':>14} | {'Quant ONNX':>14}")
    print("-" * 72)
    print(
        f"{'Disk Size (MB)':<20} | "
        f"{format_float(variants['original'].disk_size_mb):>14} | "
        f"{format_float(variants['onnx'].disk_size_mb):>14} | "
        f"{format_float(variants['quant'].disk_size_mb):>14}"
    )
    print(
        f"{'RAM Added (MB)':<20} | "
        f"{format_float(variants['original'].ram_added_mb):>14} | "
        f"{format_float(variants['onnx'].ram_added_mb):>14} | "
        f"{format_float(variants['quant'].ram_added_mb):>14}"
    )
    print("=" * 72)


def print_stage_table(stage_label, results_by_variant):
    print(f"\n{stage_label} Stage")
    print("=" * 72)
    print(f"{'Metric':<20} | {'Original':>14} | {'ONNX':>14} | {'Quant ONNX':>14}")
    print("-" * 72)
    rows = [
        ("Compute (ms)", "processing_ms"),
        ("Latency (ms)", "latency_ms"),
        ("Throughput (FPS)", "throughput_fps"),
        ("CPU Usage (%)", "cpu_usage"),
    ]
    for label, key in rows:
        print(
            f"{label:<20} | "
            f"{format_float(results_by_variant['original'][key]):>14} | "
            f"{format_float(results_by_variant['onnx'][key]):>14} | "
            f"{format_float(results_by_variant['quant'][key]):>14}"
        )
    print("=" * 72)


def print_variant_details(variants, stage_results):
    print("\nVariant Details")
    for key in ["original", "onnx", "quant"]:
        variant = variants[key]
        print(f"  {variant.label}: {variant.runtime}")
        print(f"  path: {variant.path}")
        for stage in STAGES:
            result = stage_results[key][stage.label]
            print(
                f"  {stage.label}: source={result['measurement_source']}, "
                f"detections={result['detections']}/{result['frames_processed']}, "
                f"recognitions={result['recognition_calls']}, "
                f"authorized={result['authorized_hits']}"
            )


def main():
    args = parse_args()

    args.pth_path = args.pth_path.resolve()
    args.onnx_path = args.onnx_path.resolve()
    args.quant_path = args.quant_path.resolve()

    if not args.pth_path.exists():
        raise FileNotFoundError(f"Original checkpoint not found: {args.pth_path}")
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")

    if args.detector == "yunet":
        args.detector_model = resolve_detector_model(args).resolve()
        if not args.detector_model.exists():
            detector_label = (
                "old-stage2-dropped YuNet"
                if args.yunet_variant == "old-stage2-dropped"
                else "old YuNet"
                if args.yunet_variant == "old"
                else "YuNet"
            )
            raise FileNotFoundError(
                f"{detector_label} model not found: {args.detector_model}. "
                "Provide the ONNX file at this path or pass --detector-model explicitly."
            )
    else:
        args.detector_model = None

    device = resolve_device(args.device)
    quant_device = resolve_quant_device(args.quant_device, device)

    if not args.quant_path.exists():
        if not args.auto_quantize:
            raise FileNotFoundError(
                f"Quantized ONNX model not found: {args.quant_path}. "
                "Re-run with --auto-quantize to generate it automatically."
            )
        ensure_quantized_model(args.onnx_path, args.quant_path)

    cv2.setNumThreads(max(1, args.limit_opencv_threads))

    detector = create_detector(args)
    bucket_detector_model = resolve_bucket_detector_model(args)
    bucket_detector = detector
    if bucket_detector_model is not None and bucket_detector_model != args.detector_model:
        bucket_detector = create_detector(args, bucket_detector_model)
        print(f"Using frame bucketing detector: {bucket_detector_model}")
    print(f"Collecting benchmark frames from {args.source} ...")
    raw_frames = load_benchmark_frames(args)
    frames = bucket_frames_by_detection(raw_frames, bucket_detector, args)
    print(
        f"Prepared {len(frames.face_frames)} face frames and {len(frames.no_face_frames)} no-face frames "
        f"for stage replay."
    )
    if frames.synthetic_no_face_frames:
        print("Locked-stage frames were synthesized because no natural no-face frames were found.")

    variants = build_variant_runtimes(args, device, quant_device)

    stage_results = {}
    for key in ["original", "onnx", "quant"]:
        print(f"\nBenchmarking {variants[key].label} ...")
        stage_results[key] = benchmark_variant_stages(variants[key], frames, detector, args)

    print_load_table(variants)
    for stage in STAGES:
        print_stage_table(
            stage.label,
            {
                "original": stage_results["original"][stage.label],
                "onnx": stage_results["onnx"][stage.label],
                "quant": stage_results["quant"][stage.label],
            },
        )

    print_variant_details(variants, stage_results)
    print("\nNotes:")
    if device == "cuda" or quant_device == "cuda":
        print(f"  Available ONNX Runtime providers: {ort.get_available_providers()}")
    print("  CPU Usage (%) is measured using psutil.")
    print("  Stage metrics are mean values over the practical Face Lock pipeline for the collected shared frames.")
    if frames.synthetic_no_face_frames:
        print("  Locked-stage metrics used synthetic blank frames because the capture set contained no no-face samples.")


if __name__ == "__main__":
    main()
