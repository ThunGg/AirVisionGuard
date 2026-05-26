import argparse
import ctypes
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]


DEFAULT_PTH_PATH = REPO_ROOT / "pretrained_models" / "ckpt_epoch_40.pth.tar"
DEFAULT_ONNX_PATH = REPO_ROOT / "pretrained_models" / "model.onnx"
DEFAULT_QUANT_PATH = APP_DIR / "models" / "model_quant.onnx"

MB = 1024 * 1024

KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
PSAPI = ctypes.WinDLL("psapi", use_last_error=True)


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
        description="Measure Face Lock recognition backend metrics for original, ONNX, and quantized ONNX variants."
    )
    parser.add_argument(
        "--pth-path",
        type=Path,
        default=DEFAULT_PTH_PATH,
        help="Path to the original PyTorch checkpoint.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_ONNX_PATH,
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--quant-path",
        type=Path,
        default=DEFAULT_QUANT_PATH,
        help="Path to the quantized ONNX model.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device for the original and ONNX variants.",
    )
    parser.add_argument(
        "--quant-device",
        choices=["same", "cpu", "cuda"],
        default="same",
        help="Inference device for the quantized ONNX variant.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size.")
    parser.add_argument("--iterations", type=int, default=100, help="Timed inference iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before timing.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for CUDA execution and nvidia-smi power sampling.",
    )
    parser.add_argument(
        "--power-sample-ms",
        type=int,
        default=100,
        help="GPU power sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        help="Backbone name for the original checkpoint. Use 'auto' to read the checkpoint arch field.",
    )
    parser.add_argument("--feature-dim", type=int, default=512, help="Embedding size for the original model.")
    parser.add_argument("--input-size", type=int, default=112, help="Input height/width for dummy inference.")
    parser.add_argument(
        "--feat-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the original framework model uses feature batch norm.",
    )
    parser.add_argument(
        "--auto-quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the quantized ONNX model automatically when it is missing.",
    )
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
        raise RuntimeError(
            "Could not infer the backbone from the checkpoint. Please pass --backbone explicitly."
        )
    return fallback_backbone


def load_backbone_factory(backbone_name: str):
    if backbone_name not in {"mobilefacenet", "mobilefacenet_large"}:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}' for this app benchmark. "
            "Use the Face Lock default MobileFaceNet checkpoint or extend this script with the required backbone factory."
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


def load_original_model(args, device: str):
    backbone = infer_backbone(args.pth_path, args.backbone)
    model = OriginalInferenceModel(
        backbone_name=backbone,
        feature_dim=args.feature_dim,
        input_size=args.input_size,
        feat_bn=args.feat_bn,
    )
    checkpoint, missing, unexpected = load_checkpoint_state(model, args.pth_path)
    if missing:
        print(f"[original] Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"[original] Unexpected checkpoint keys: {unexpected}")
    model.eval()
    model.to(device)
    arch_name = checkpoint.get("arch", backbone) if isinstance(checkpoint, dict) else backbone
    return model, arch_name


def build_session(path: Path, requested_device: str, gpu_index: int):
    providers = ["CPUExecutionProvider"]
    provider_used = "CPUExecutionProvider"
    if requested_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        cuda_options = {"device_id": gpu_index}
        providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
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

    # Pack external-data ONNX exports into a single local file first so the
    # quantizer operates entirely within the app workspace.
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


class NvidiaSmiPowerSampler:
    def __init__(self, gpu_index: int, interval_s: float):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = self._probe()

    def _probe(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _read_power_w(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu_index),
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

        if result.returncode != 0:
            return None

        first_line = result.stdout.strip().splitlines()[0].strip()
        try:
            return float(first_line)
        except ValueError:
            return None

    def _run(self):
        while not self._stop.is_set():
            power_w = self._read_power_w()
            if power_w is not None:
                self.samples.append(power_w)
            time.sleep(self.interval_s)

    def start(self):
        if not self.available:
            return
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available:
            return
        self._stop.set()
        self._thread.join()

    @property
    def average_power_w(self):
        if not self.samples:
            return None
        return float(np.mean(self.samples))


def measure_torch_latency_ms(model, input_tensor, iterations, warmup, device, gpu_index, power_sample_s):
    device_obj = torch.device(device)
    input_on_device = input_tensor.to(device_obj)

    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(input_on_device, extract_mode=True)
    if device == "cuda":
        torch.cuda.synchronize(device_obj)

    power_sampler = None
    if device == "cuda":
        power_sampler = NvidiaSmiPowerSampler(gpu_index=gpu_index, interval_s=power_sample_s)
        power_sampler.start()

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.inference_mode():
            _ = model(input_on_device, extract_mode=True)
    if device == "cuda":
        torch.cuda.synchronize(device_obj)
    elapsed_s = time.perf_counter() - start

    avg_power_w = None
    if power_sampler is not None:
        power_sampler.stop()
        avg_power_w = power_sampler.average_power_w

    return (elapsed_s / iterations) * 1000.0, avg_power_w


def measure_ort_latency_ms(session, input_tensor, iterations, warmup, gpu_index, power_sample_s):
    provider = session.get_providers()[0] if session.get_providers() else "CPUExecutionProvider"
    input_name = session.get_inputs()[0].name
    is_cuda = provider == "CUDAExecutionProvider" and torch.cuda.is_available()

    if is_cuda:
        device = torch.device(f"cuda:{gpu_index}")
        input_on_device = input_tensor.to(device).contiguous()
        output_shape = []
        for dim in session.get_outputs()[0].shape:
            if isinstance(dim, str) or dim is None or dim <= 0:
                output_shape.append(input_tensor.shape[0])
            else:
                output_shape.append(dim)
        output_tensor = torch.empty(tuple(output_shape), dtype=torch.float32, device=device).contiguous()
        io_binding = session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=gpu_index,
            element_type=np.float32,
            shape=tuple(input_on_device.shape),
            buffer_ptr=input_on_device.data_ptr(),
        )
        io_binding.bind_output(
            name=session.get_outputs()[0].name,
            device_type="cuda",
            device_id=gpu_index,
            element_type=np.float32,
            shape=tuple(output_tensor.shape),
            buffer_ptr=output_tensor.data_ptr(),
        )

        for _ in range(warmup):
            session.run_with_iobinding(io_binding)
        torch.cuda.synchronize(device)

        power_sampler = NvidiaSmiPowerSampler(gpu_index=gpu_index, interval_s=power_sample_s)
        power_sampler.start()

        start = time.perf_counter()
        for _ in range(iterations):
            session.run_with_iobinding(io_binding)
        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start

        power_sampler.stop()
        return (elapsed_s / iterations) * 1000.0, power_sampler.average_power_w

    input_array = input_tensor.cpu().numpy()
    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: input_array})
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / iterations) * 1000.0, None


def format_float(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def print_table(results):
    headers = ["Metric", "Original", "ONNX", "Quant ONNX"]
    rows = [
        ("Disk Size (MB)", "disk_size_mb"),
        ("RAM Added (MB)", "ram_added_mb"),
        ("Latency (ms)", "latency_ms"),
        ("Throughput (FPS)", "throughput_fps"),
        ("GPU Power (W)", "gpu_power_w"),
    ]

    widths = [20, 14, 14, 14]
    print("\n" + "=" * 72)
    print(
        f"{headers[0]:<{widths[0]}} | "
        f"{headers[1]:>{widths[1]}} | "
        f"{headers[2]:>{widths[2]}} | "
        f"{headers[3]:>{widths[3]}}"
    )
    print("-" * 72)
    for label, key in rows:
        print(
            f"{label:<{widths[0]}} | "
            f"{format_float(results['original'].get(key)):>{widths[1]}} | "
            f"{format_float(results['onnx'].get(key)):>{widths[2]}} | "
            f"{format_float(results['quant'].get(key)):>{widths[3]}}"
        )
    print("=" * 72)


def print_variant_details(results):
    print("Variant details:")
    print(f"  original checkpoint: {results['original']['path']}")
    print(f"  original device:     {results['original']['runtime']}")
    print(f"  ONNX path:           {results['onnx']['path']}")
    print(f"  ONNX provider:       {results['onnx']['runtime']}")
    print(f"  quant ONNX path:     {results['quant']['path']}")
    print(f"  quant provider:      {results['quant']['runtime']}")


def main():
    args = parse_args()
    args.pth_path = args.pth_path.resolve()
    args.onnx_path = args.onnx_path.resolve()
    args.quant_path = args.quant_path.resolve()

    if not args.pth_path.exists():
        raise FileNotFoundError(f"Original checkpoint not found: {args.pth_path}")
    if not args.onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")

    device = resolve_device(args.device)
    quant_device = resolve_quant_device(args.quant_device, device)
    power_sample_s = max(0.01, args.power_sample_ms / 1000.0)

    if not args.quant_path.exists():
        if not args.auto_quantize:
            raise FileNotFoundError(
                f"Quantized ONNX model not found: {args.quant_path}. "
                "Re-run with --auto-quantize to generate it automatically."
            )
        ensure_quantized_model(args.onnx_path, args.quant_path)

    dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, dtype=torch.float32)
    results = {
        "original": {"path": str(args.pth_path)},
        "onnx": {"path": str(args.onnx_path)},
        "quant": {"path": str(args.quant_path)},
    }

    print(f"Benchmarking Face Lock recognition variants with batch_size={args.batch_size}, iterations={args.iterations}")
    print(f"Primary device: {device}")
    print(f"Quantized device: {quant_device}")

    original_model, original_ram = measure_ram_added(lambda: load_original_model(args, device))
    original_runner, arch_name = original_model
    original_latency_ms, original_power_w = measure_torch_latency_ms(
        model=original_runner,
        input_tensor=dummy_input,
        iterations=args.iterations,
        warmup=args.warmup,
        device=device,
        gpu_index=args.gpu_index,
        power_sample_s=power_sample_s,
    )
    results["original"].update(
        {
            "runtime": f"PyTorch on {device} ({arch_name})",
            "disk_size_mb": get_file_size_mb(args.pth_path),
            "ram_added_mb": original_ram,
            "latency_ms": original_latency_ms,
            "throughput_fps": (1000.0 / original_latency_ms) * args.batch_size if original_latency_ms > 0 else None,
            "gpu_power_w": original_power_w,
        }
    )

    onnx_session, onnx_ram = measure_ram_added(lambda: build_session(args.onnx_path, device, args.gpu_index))
    onnx_runner, onnx_provider = onnx_session
    onnx_latency_ms, onnx_power_w = measure_ort_latency_ms(
        session=onnx_runner,
        input_tensor=dummy_input,
        iterations=args.iterations,
        warmup=args.warmup,
        gpu_index=args.gpu_index,
        power_sample_s=power_sample_s,
    )
    results["onnx"].update(
        {
            "runtime": onnx_provider,
            "disk_size_mb": get_file_size_mb(args.onnx_path),
            "ram_added_mb": onnx_ram,
            "latency_ms": onnx_latency_ms,
            "throughput_fps": (1000.0 / onnx_latency_ms) * args.batch_size if onnx_latency_ms > 0 else None,
            "gpu_power_w": onnx_power_w,
        }
    )

    quant_session, quant_ram = measure_ram_added(lambda: build_session(args.quant_path, quant_device, args.gpu_index))
    quant_runner, quant_provider = quant_session
    quant_latency_ms, quant_power_w = measure_ort_latency_ms(
        session=quant_runner,
        input_tensor=dummy_input,
        iterations=args.iterations,
        warmup=args.warmup,
        gpu_index=args.gpu_index,
        power_sample_s=power_sample_s,
    )
    results["quant"].update(
        {
            "runtime": quant_provider,
            "disk_size_mb": get_file_size_mb(args.quant_path),
            "ram_added_mb": quant_ram,
            "latency_ms": quant_latency_ms,
            "throughput_fps": (1000.0 / quant_latency_ms) * args.batch_size if quant_latency_ms > 0 else None,
            "gpu_power_w": quant_power_w,
        }
    )

    print_table(results)
    print_variant_details(results)
    if device == "cuda" or quant_device == "cuda":
        print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
        print("GPU power is sampled with nvidia-smi while each timed loop runs.")
    else:
        print("GPU power is reported as N/A when running on CPU.")


if __name__ == "__main__":
    main()
