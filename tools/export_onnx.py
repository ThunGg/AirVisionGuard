import torch
import torch.nn as nn
import argparse
import os
import yaml
import sys

# Add project root to path to allow imports from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models
from utils import load_state, log

class ONNXInferenceModel:
    """Wrapper for utilizing onnxruntime for inference."""
    def __init__(self, model_path, use_cuda=True):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        """Perform inference.
        
        Args:
            x (np.ndarray or torch.Tensor): Input image tensor.
            
        Returns:
            np.ndarray: Extracted features.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        ort_inputs = {self.input_name: x}
        ort_outs = self.session.run(None, ort_inputs)
        return ort_outs[0]

def export_onnx(config_path, load_path, output_path, input_size=(1, 3, 112, 112), quantize=False):
    # 1. Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Instantiate model
    log("Creating model for export...")
    model = models.MultiTaskWithLoss(
        backbone=config['model']['backbone'],
        num_classes=None, # Inference mode (feature extraction)
        feature_dim=config['model']['feature_dim'],
        spatial_size=config['model']['input_size'],
        arc_fc=config['model']['arc_fc'],
        feat_bn=config['model']['feat_bn']
    )

    # 3. Load weights
    log(f"Loading weights from {load_path}...")
    load_state(load_path, model)
    model.eval()

    # 4. Create a wrapper for ONNX export to handle the forward pass correctly
    # The framework's forward() has many arguments, we simplify it for ONNX.
    class OnnxExportWrapper(nn.Module):
        def __init__(self, model):
            super(OnnxExportWrapper, self).__init__()
            self.model = model
        
        def forward(self, x):
            # Equivalent to model(x, extract_mode=True)
            return self.model(x, extract_mode=True)

    export_model = OnnxExportWrapper(model)
    export_model.eval()

    # 5. Export
    dummy_input = torch.randn(*input_size)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    log(f"Exporting to ONNX (opset=12, dynamic batch size)...")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    log(f"SUCCESS: Model exported to {output_path}")
    
    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        log("Verifying exported model with ONNX Runtime...")
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        
        with torch.no_grad():
            torch_out = export_model(dummy_input).numpy()
        
        # Compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)
        log("Verification successful! Exported model results match PyTorch.")
    except ImportError:
        log("Warning: onnxruntime not installed, skipping verification.")
    except Exception as e:
        log(f"Verification failed: {e}")

    # 6. Quantization
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = output_path.replace(".onnx", "_quant.onnx")
            log(f"Quantizing model to {quant_path}...")
            quantize_dynamic(
                output_path,
                quant_path,
                weight_type=QuantType.QUInt8
            )
            log(f"SUCCESS: Quantized model saved to {quant_path}")
        except ImportError:
            log("Error: onnxruntime.quantization not found. Please install onnxruntime.")
        except Exception as e:
            log(f"Quantization failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export face recognition model to ONNX')
    parser.add_argument('--config', type=str, required=True, help='Path to model config yaml')
    parser.add_argument('--load-path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output-path', type=str, default='model.onnx', help='Output path for .onnx file')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for dummy input')
    parser.add_argument('--quantize', action='store_true', help='Perform dynamic quantization')
    
    args = parser.parse_args()
    
    # Extract spatial size from config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    spatial_size = config['model']['input_size']
    
    if isinstance(spatial_size, int):
        dummy_shape = (args.batch_size, 3, spatial_size, spatial_size)
    else:
        dummy_shape = (args.batch_size, 3, spatial_size[0], spatial_size[1])

    export_onnx(args.config, args.load_path, args.output_path, dummy_shape, quantize=args.quantize)

