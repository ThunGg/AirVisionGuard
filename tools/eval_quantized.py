import argparse
import os
import sys
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add parent directory to path to allow importing models, utils, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import BinDataset, FileListDataset, GivenSizeSampler
from utils import AverageMeter, log, normalize
from evaluation.verify import evaluate
from evaluation.megaface import test_megaface
from tools.load_quantized import load_quantized_model, ArgObj

def _build_loader_kwargs(num_workers):
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    return loader_kwargs

def extract(ext_loader, model, num, silent=False):
    batch_time = AverageMeter(9999999)
    data_time = AverageMeter(9999999)
    model.eval()
    features = []

    start = time.time()
    end = time.time()
    with torch.inference_mode():
        for i, input in enumerate(ext_loader):
            data_time.update(time.time() - end)
            # For dynamic quantization, it's typically best to keep input on CPU 
            # if the model is on CPU, otherwise move to the same device as model.
            input = input.to(next(model.parameters()).device)
            
            output = model(input, extract_mode=True)
            features.append(output.detach().float().cpu().numpy())
            
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0 and not silent:
                log("Extracting: {0}/{1}\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})".format(
                    i, len(ext_loader), batch_time=batch_time, data_time=data_time))

    features = np.concatenate(features, axis=0)[:num, :]
    if not silent:
        log("Extracting Done. Total time: {}".format(time.time() - start))
    return features

def evaluation(test_loader, model, num, benchmark, nfolds=10, labels=None):
    log(f"Evaluating on {benchmark}...")
    features = extract(test_loader, model, num, silent=False)
    
    if benchmark == "megaface":
        r = test_megaface(features)
        log(' * Megaface: 1e-6 [{}], 1e-5 [{}], 1e-4 [{}]'.format(r[-1], r[-2], r[-3]))
        return r[-1]
    else:
        features = normalize(features)
        if labels is None:
            raise ValueError("Verification labels are required for non-MegaFace evaluation")
        lbs = np.asarray(labels).astype(bool)
        _, _, acc, val, val_std, far = evaluate(
            features, lbs, nrof_folds=nfolds, distance_metric=0)
    
        log(" * {}: accuracy: {:.4f}({:.4f})".format(benchmark, acc.mean(), acc.std()))
        return acc.mean()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Quantized Model')
    parser.add_argument('--config', type=str, required=True, help='Path to training config.yaml')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth.tar checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cpu', help='Device to evaluate on (quantized models usually run best on CPU)')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load config
    log(f"Loading config from {args.config}")
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = ArgObj(config_dict)

    # Load and quantize model
    model = load_quantized_model(args.config, args.model_path, device=args.device)

    # Setup datasets based on config.test
    test_datasets = []
    benchmarks = getattr(config.test, 'benchmark', [])
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]

    if not benchmarks:
        log("No validation datasets (benchmarks) specified in config.test.benchmark.")
        return

    input_size = config.model.input_size
    batch_size = config.test.batch_size

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for tb in benchmarks:
        if tb == 'megaface':
            test_datasets.append(FileListDataset(
                config.test.megaface_list,
                config.test.megaface_root, 
                transform
            ))
        else:
            test_datasets.append(BinDataset(
                "{}/{}.bin".format(config.test.test_root, tb),
                transform
            ))

    test_samplers = [GivenSizeSampler(td,
        total_size=int(np.ceil(len(td) / float(batch_size)) * batch_size),
        sequential=True, silent=True) for td in test_datasets]
        
    test_loaders = [DataLoader(
        td, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=ts,
        **_build_loader_kwargs(args.workers))
        for td, ts in zip(test_datasets, test_samplers)]

    nfolds = getattr(config.test, 'nfolds', 10)

    for tb, tl, td in zip(benchmarks, test_loaders, test_datasets):
        log(f"--- Starting evaluation for {tb} ---")
        evaluation(
            test_loader=tl, 
            model=model, 
            num=len(td), 
            benchmark=tb,
            nfolds=nfolds,
            labels=getattr(td, 'lbs', None)
        )
    log("Evaluation of quantized model finished.")

if __name__ == '__main__':
    main()
