import argparse
import os
import torch
import yaml
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import models
from utils import load_state, normalize, log

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to training config.yaml')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth.tar checkpoint')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-path', type=str, default='embedding.bin', help='Path to save extracted embedding')
    return parser.parse_args()

class ArgObj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ArgObj(v))
            else:
                setattr(self, k, v)

def main():
    args = get_args()

    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert config to object (similar to main.py logic)
    config = ArgObj(config_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # 1. Initialize Model
    # We use num_classes=None because we only care about the backbone for inference
    model = models.MultiTaskWithLoss(
        backbone=config.model.backbone,
        num_classes=None,
        feature_dim=config.model.feature_dim,
        spatial_size=config.model.input_size,
        feat_bn=getattr(config.model, 'feat_bn', False)
    )

    # 2. Load Weights
    load_state(args.model_path, model)
    model = model.to(device)
    model.eval()

    # 3. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize(config.model.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if not os.path.exists(args.image_path):
        log(f"Error: Image not found at {args.image_path}")
        return

    img = Image.open(args.image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Extract Embedding
    with torch.no_grad():
        # MultiTaskWithLoss forward has an extract_mode parameter
        embedding = model(img_tensor, extract_mode=True)
        embedding = embedding.cpu().numpy()
        embedding = normalize(embedding) # L2 Normalize

    # 5. Save/Output
    log(f"Embedding extracted successfully. Shape: {embedding.shape}")
    embedding.tofile(args.output_path)
    log(f"Embedding saved to {args.output_path}")

if __name__ == '__main__':
    main()
