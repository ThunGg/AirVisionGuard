import argparse
import os
import sys
import torch
import yaml
import logging
import psutil

# Add parent directory to path to allow importing models and utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
from utils import load_state, log

# Map of supported quantization dtype strings to their torchao config constructors.
# Lazily imported to avoid hard dependency at module level.
QUANT_DTYPE_MAP = {
    'int8_dynamic': 'Int8DynamicActivationInt8WeightConfig',
    'int8_weight_only': 'Int8WeightOnlyConfig',
    'int4_weight_only': 'Int4WeightOnlyConfig',
}

class ArgObj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ArgObj(v))
            else:
                setattr(self, k, v)

def _get_quant_config(quant_dtype):
    """
    Returns a torchao quantization config object for the given dtype string.

    Args:
        quant_dtype (str): One of the keys in QUANT_DTYPE_MAP.

    Returns:
        A torchao quantization config instance.
    """
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        Int4WeightOnlyConfig,
    )

    config_cls_name = QUANT_DTYPE_MAP.get(quant_dtype)
    if config_cls_name is None:
        raise ValueError(
            f"Unsupported quant_dtype '{quant_dtype}'. "
            f"Choose from: {list(QUANT_DTYPE_MAP.keys())}"
        )

    config_cls = {
        'Int8DynamicActivationInt8WeightConfig': Int8DynamicActivationInt8WeightConfig,
        'Int8WeightOnlyConfig': Int8WeightOnlyConfig,
        'Int4WeightOnlyConfig': Int4WeightOnlyConfig,
    }[config_cls_name]

    return config_cls()


def load_quantized_model(config_path, model_path, device='cpu',
                         quantize=True, quant_dtype='int8_dynamic'):
    """
    Loads a trained model and optionally applies quantization via torchao.

    Args:
        config_path (str): Path to the model configuration YAML file.
        model_path (str): Path to the trained .pth.tar checkpoint.
        device (str or torch.device): Device to load the model on.
        quantize (bool): Whether to apply quantization. Defaults to True.
        quant_dtype (str): Quantization scheme to use. One of:
            - 'int8_dynamic'     : INT8 dynamic activation + INT8 weight
            - 'int8_weight_only' : INT8 weight-only quantization
            - 'int4_weight_only' : INT4 weight-only quantization (group_size=32)
            Defaults to 'int8_dynamic'.

    Returns:
        torch.nn.Module: The (optionally quantized) PyTorch model.
    """
    log(f"Loading config from {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = ArgObj(config_dict)

    log(f"Initializing model backbone: {config.model.backbone}")
    model = models.MultiTaskWithLoss(
        backbone=config.model.backbone,
        num_classes=None,
        feature_dim=config.model.feature_dim,
        spatial_size=config.model.input_size,
        feat_bn=getattr(config.model, 'feat_bn', False)
    )

    log(f"Loading checkpoint from {model_path}")
    load_state(model_path, model)

    model = model.to(device)
    model.eval()

    if quantize:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        log(f"RAM usage before quantization: {mem_before:.2f} MB")

        log(f"Applying torchao quantization with scheme: {quant_dtype}")
        try:
            from torchao.quantization import quantize_
            qconfig = _get_quant_config(quant_dtype)
            quantize_(model, qconfig)
            log("Model quantization applied successfully.")
        except Exception as e:
            log(f"Failed to apply quantization: {e}")

        mem_after = process.memory_info().rss / (1024 * 1024)
        log(f"RAM usage after quantization: {mem_after:.2f} MB")
        log(f"RAM usage difference: {mem_after - mem_before:.2f} MB")
    else:
        log("Skipping quantization as quantize=False.")

    return model

def main():
    parser = argparse.ArgumentParser(description='Load trained model and apply quantization')
    parser.add_argument('--config', type=str, required=True, help='Path to training config.yaml')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth.tar checkpoint')
    parser.add_argument('--output-path', type=str, default='quantized_model.pth', help='Path to save the quantized model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run quantization on (default: cpu)')
    parser.add_argument('--quant-dtype', type=str, default='int8_dynamic',
                        choices=list(QUANT_DTYPE_MAP.keys()),
                        help='Quantization scheme to apply (default: int8_dynamic)')
    args = parser.parse_args()

    # Set up logging if run as main
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    log(f"Starting quantization process on device: {args.device}")

    quantized_model = load_quantized_model(
        args.config, args.model_path,
        device=args.device, quant_dtype=args.quant_dtype
    )
    
    log(f"Saving quantized model state dict to {args.output_path}")
    torch.save(quantized_model.state_dict(), args.output_path)
    log("Quantization utility finished successfully.")

if __name__ == '__main__':
    main()
