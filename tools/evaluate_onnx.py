import torch
import numpy as np
import time
import os
import argparse
import yaml
import sys
import psutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from utils import load_state, log

def get_file_size(path):
    """Get file size in MB."""
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) / (1024 * 1024)

def measure_ram(load_func):
    """Estimate RAM usage by measuring process RSS before and after model loading."""
    process = psutil.Process(os.getpid())
    # Garbage collect before measurement
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    mem_before = process.memory_info().rss / (1024 * 1024)
    model = load_func()
    gc.collect()
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    return model, max(0, mem_after - mem_before)

def measure_speed(model, dummy_input, iterations=100, device='cuda'):
    """Measure inference speed in milliseconds."""
    is_torch = isinstance(model, torch.nn.Module)
    
    # Warmup
    warmup_iters = 10
    for _ in range(warmup_iters):
        if is_torch:
            with torch.inference_mode():
                _ = model(dummy_input.to(device), extract_mode=True)
        else: # ONNX
            _ = model.run(None, {'input': dummy_input.numpy()})

    if device == 'cuda' and is_torch:
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(iterations):
        if is_torch:
            with torch.inference_mode():
                _ = model(dummy_input.to(device), extract_mode=True)
        else: # ONNX
            _ = model.run(None, {'input': dummy_input.numpy()})
    
    if device == 'cuda' and is_torch:
        torch.cuda.synchronize()
        
    end_time = time.time()
    return (end_time - start_time) / iterations * 1000

def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch and ONNX model performance')
    parser.add_argument('--config', type=str, required=True, help='Model config file')
    parser.add_argument('--pth-path', type=str, required=True, help='Path to PyTorch .pth file')
    parser.add_argument('--onnx-path', type=str, required=True, help='Path to exported .onnx file')
    parser.add_argument('--quant-path', type=str, default=None, help='Path to quantized .onnx file (optional)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for speed test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    # Import ORT here so it's not a hard dependency for just looking at the script
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime is required for this script. Install it with 'pip install onnxruntime-gpu' or 'onnxruntime'.")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    spatial_size = config['model']['input_size']
    if isinstance(spatial_size, int):
        input_shape = (args.batch_size, 3, spatial_size, spatial_size)
    else:
        input_shape = (args.batch_size, 3, spatial_size[0], spatial_size[1])
    
    dummy_input = torch.randn(*input_shape)

    log(f"\nEvaluating performance on device: {args.device}")
    log("-" * 70)

    # --- PyTorch Model ---
    log("Loading PyTorch model...")
    def load_pytorch():
        model = models.MultiTaskWithLoss(
            backbone=config['model']['backbone'],
            num_classes=None,
            feature_dim=config['model']['feature_dim'],
            spatial_size=config['model']['input_size'],
            arc_fc=config['model']['arc_fc'],
            feat_bn=config['model']['feat_bn']
        )
        load_state(args.pth_path, model)
        model.to(args.device)
        model.eval()
        return model

    pytorch_model, pt_ram = measure_ram(load_pytorch)
    pt_size = get_file_size(args.pth_path)
    log("Measuring PyTorch speed...")
    pt_speed = measure_speed(pytorch_model, dummy_input, iterations=args.iterations, device=args.device)

    # --- ONNX Model ---
    log("Loading ONNX model...")
    def load_onnx(path):
        providers = ['CPUExecutionProvider']
        if args.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        session = ort.InferenceSession(path, providers=providers)
        return session

    onnx_session, onnx_ram = measure_ram(lambda: load_onnx(args.onnx_path))
    onnx_size = get_file_size(args.onnx_path)
    log("Measuring ONNX speed...")
    onnx_speed = measure_speed(onnx_session, dummy_input, iterations=args.iterations, device=args.device)

    # --- Quantized ONNX Model ---
    quant_session, quant_ram, quant_size, quant_speed = None, None, None, None
    if args.quant_path and os.path.exists(args.quant_path):
        log("Loading Quantized ONNX model...")
        # Quantized models are typically best on CPU
        quant_session, quant_ram = measure_ram(lambda: load_onnx(args.quant_path))
        quant_size = get_file_size(args.quant_path)
        log("Measuring Quantized ONNX speed...")
        quant_speed = measure_speed(quant_session, dummy_input, iterations=args.iterations, device=args.device)

    # --- Results ---
    log("\n" + "="*85)
    header = f"{'Metric':<20} | {'PyTorch':<12} | {'ONNX':<12}"
    if quant_session:
        header += f" | {'Quant ONNX':<12}"
    header += " | {'Change (ONNX)'}"
    log(header)
    log("-" * 85)
    
    def format_change(old, new):
        if old == 0: return "N/A"
        ratio = new / old
        diff = (ratio - 1) * 100
        return f"{ratio:.2f}x ({diff:+.1f}%)"

    def log_metric(name, pt_val, onnx_val, quant_val):
        row = f"{name:<20} | {pt_val:>12.2f} | {onnx_val:>12.2f}"
        if quant_session:
            row += f" | {quant_val:>12.2f}"
        row += f" | {format_change(pt_val, onnx_val)}"
        log(row)

    log_metric('Disk Size (MB)', pt_size, onnx_size, quant_size)
    log_metric('RAM Added (MB)', pt_ram, onnx_ram, quant_ram)
    log_metric('Latency (ms)', pt_speed, onnx_speed, quant_speed)
    
    # Calculate throughput
    pt_fps = (1000 / pt_speed) * args.batch_size if pt_speed > 0 else 0
    onnx_fps = (1000 / onnx_speed) * args.batch_size if onnx_speed > 0 else 0
    quant_fps = (1000 / quant_speed) * args.batch_size if quant_speed and quant_speed > 0 else 0
    
    row = f"{'Throughput (FPS)':<20} | {pt_fps:>12.2f} | {onnx_fps:>12.2f}"
    if quant_session:
        row += f" | {quant_fps:>12.2f}"
    row += f" | {format_change(pt_fps, onnx_fps)}"
    log(row)
    
    log("="*85)
    log(f"Inference device: {args.device}")
    if args.device == 'cuda':
        log(f"Available ORT Providers: {ort.get_available_providers()}")


if __name__ == '__main__':
    main()
