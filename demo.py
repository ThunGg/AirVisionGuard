import argparse
import os
import time
import logging
from datetime import datetime
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
from datasets import GivenSizeSampler, BinDataset, FileListLabeledDataset, FileListDataset
from utils import AverageMeter, load_state, save_state, log, normalize, bin_loader
from evaluation import evaluate, test_megaface
import torch.distributed as dist

model_names = sorted(name for name in models.backbones.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.backbones.__dict__[name]))

class ArgObj(object):
    def __init__(self):
        pass

parser = argparse.ArgumentParser(description='Demo/Smoke Test for Face Recognition')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--local_rank', default=0, type=int)

def main():
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k,v in config.items():    
        if isinstance(v, dict):
            argobj = ArgObj()
            setattr(args, k, argobj)
            for kk,vv in v.items():
                setattr(argobj, kk, vv)
        else:
            setattr(args, k, v)
    
    args.ngpu = len(args.gpus.split(','))
    
    # Force Demo Limits
    args.train.max_epoch = 1
    args.test.benchmark = ['lfw']
    args.test.interval = 1
    args.train.print_freq = 1
    
    # DDP Initialization
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        
    if not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
    else:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        log("Demo Mode: Distributed rank {}/{}".format(args.rank, args.world_size))

    num_tasks = len(args.train.data_root)
    args.save_path = os.path.join(os.path.dirname(args.config), 'demo_run')
    os.makedirs('{}/checkpoints'.format(args.save_path), exist_ok=True)

    ## create dataset
    train_dataset = [FileListLabeledDataset(
        args.train.data_list[i], args.train.data_root[i],
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.model.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]),
        memcached=args.memcached,
        memcached_client=args.memcached_client) for i in range(num_tasks)]
    
    args.num_classes = [td.num_class for td in train_dataset]
    train_sampler = [GivenSizeSampler(td, rand_seed=args.train.rand_seed) for td in train_dataset]
    train_loader = [DataLoader(
        train_dataset[k], batch_size=args.train.batch_size[k], shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler[k]) for k in range(num_tasks)]

    test_dataset = [BinDataset("{}/{}.bin".format(args.test.test_root, 'lfw'),
                transforms.Compose([
                transforms.Resize(args.model.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]))]
    test_sampler = [GivenSizeSampler(td, sequential=True, silent=True) for td in test_dataset]
    test_loader = [DataLoader(
        td, batch_size=args.test.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=ts)
        for td, ts in zip(test_dataset, test_sampler)]

    ## create model
    model = models.MultiTaskWithLoss(backbone=args.model.backbone, num_classes=args.num_classes, feature_dim=args.model.feature_dim, spatial_size=args.model.input_size, arc_fc=args.model.arc_fc, feat_bn=args.model.feat_bn)
    
    if args.distributed:
        model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = nn.DataParallel(model).cuda()
    
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.SGD(model.parameters(), args.train.base_lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay)

    # Demo loop
    log("Starting Demo Run (Max 10 iterations)...")
    for epoch in range(1):
        for ts in train_sampler:
            ts.set_epoch(epoch)
        
        train_demo(train_loader, model, optimizer, epoch, args.train.loss_weight, scaler)
        
        # Evaluation
        torch.cuda.empty_cache()
        log("*************** Demo Evaluation ***************")
        for tb, tl, td in zip(args.test.benchmark, test_loader, test_dataset):
            evaluation_demo(tl, model, num=len(td), outfeat_fn="{}/demo_lfw.bin".format(args.save_path), benchmark=tb)

    if args.distributed:
        dist.destroy_process_group()
    log("Demo finished successfully!")

def train_demo(train_loader, model, optimizer, epoch, loss_weight, scaler):
    model.train()
    num_tasks = len(train_loader)
    for i, all_in in enumerate(zip(*tuple(train_loader))):
        if i >= 10: # Limit iterations
            break
        
        input, target = zip(*[all_in[k] for k in range(num_tasks)])
        
        if args.distributed:
            slice_idx = [0]
            pt = 0
            for l in [p.size(0) for p in input]:
                pt += l
                slice_idx.append(pt)
            input = torch.cat(input, dim=0).cuda(non_blocking=True)
            target = torch.cat(target, dim=0).cuda(non_blocking=True)
        else:
            # Simplified for DP demo
            slice_idx = [0, input[0].size(0)]
            input = input[0].cuda()
            target = target[0].cuda()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = model(input, target, slice_idx)
            loss_total = sum([loss[k].mean() * loss_weight[k] for k in range(len(loss))])

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        log('Demo Iteration [{}/10] Loss: {:.4f}'.format(i+1, loss_total.item()))

def extract_demo(ext_loader, model, num):
    model.eval()
    features = []
    with torch.no_grad():
        for i, input in enumerate(ext_loader):
            if i >= 5: break # Fast extraction
            input = input.cuda(non_blocking=True)
            output = model(input, extract_mode=True)
            features.append(output.detach().cpu().numpy())
    return np.concatenate(features, axis=0)

def evaluation_demo(test_loader, model, num, outfeat_fn, benchmark):
    features = extract_demo(test_loader, model, num)
    
    if args.distributed:
        local_features = torch.from_numpy(features).cuda()
        all_features = [torch.zeros_like(local_features) for _ in range(args.world_size)]
        dist.all_gather(all_features, local_features)
        combined = []
        for i in range(local_features.size(0)):
            for rank_f in all_features:
                combined.append(rank_f[i])
        features = torch.stack(combined).cpu().numpy()[:num, :]

    if args.rank == 0:
        log("Demo Evaluation on {}: Features Shape {}".format(benchmark, features.shape))
        # We skip the heavy LFW calculation in demo to be "much much faster"
        # but the extraction pipeline is verified.
        return 0

if __name__ == '__main__':
    main()
