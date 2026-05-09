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

class ArgObj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ArgObj(v))
            else:
                setattr(self, k, v)

def load_quantized_model(config_path, model_path, device='cpu', qconfig_spec=None, quantize=True):
    """
    Loads a trained model and optionally applies dynamic quantization.
    
    Args:
        config_path (str): Path to the model configuration YAML file.
        model_path (str): Path to the trained .pth.tar checkpoint.
        device (str or torch.device): Device to load the model on.
        qconfig_spec (set or dict, optional): Specification for layers to quantize. 
                                              Defaults to {torch.nn.Linear}.
        quantize (bool): Whether to apply dynamic quantization. Defaults to True.
    
    Returns:
        torch.nn.Module: The quantized PyTorch model.
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
    if quantize and str(device) != 'cpu':
        log("Warning: Dynamic quantization requires CPU. Forcing device to 'cpu'.")
        device = 'cpu'

    model = model.to(device)
    model.eval()

    if quantize:
        if qconfig_spec is None:
            qconfig_spec = {torch.nn.Linear}

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        log(f"RAM usage before quantization: {mem_before:.2f} MB")

        log(f"Applying dynamic quantization to: {qconfig_spec}")
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, qconfig_spec, dtype=torch.qint8
            )
            log("Model quantization applied successfully.")
        except Exception as e:
            log(f"Failed to apply dynamic quantization: {e}")
            quantized_model = model

        mem_after = process.memory_info().rss / (1024 * 1024)
        log(f"RAM usage after quantization: {mem_after:.2f} MB")
        log(f"RAM usage difference: {mem_after - mem_before:.2f} MB")
    else:
        log("Skipping quantization as quantize=False.")
        quantized_model = model

    return quantized_model

def main():
    parser = argparse.ArgumentParser(description='Load trained model and apply quantization')
    parser.add_argument('--config', type=str, required=True, help='Path to training config.yaml')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth.tar checkpoint')
    parser.add_argument('--output-path', type=str, default='quantized_model.pth', help='Path to save the quantized model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run quantization on (default: cpu)')
    args = parser.parse_args()

    # Set up logging if run as main
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    log(f"Starting quantization process on device: {args.device}")

    quantized_model = load_quantized_model(args.config, args.model_path, device=args.device)
    
    log(f"Saving quantized model state dict to {args.output_path}")
    torch.save(quantized_model.state_dict(), args.output_path)
    log("Quantization utility finished successfully.")

if __name__ == '__main__':
    main()
