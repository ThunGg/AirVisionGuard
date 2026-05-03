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
from utils import AverageMeter, load_state, save_state, log, normalize
from evaluation import evaluate, test_megaface
import torch.distributed as dist

model_names = sorted(name for name in models.backbones.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.backbones.__dict__[name]))

class ArgObj(object):
    def __init__(self):
        pass

parser = argparse.ArgumentParser(description='Multi-Task Face Recognition Training')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--save-path', default='', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--extract', action='store_true')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--demo', action='store_true', help='run in demo mode (fast verification)')
parser.add_argument('opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER)


def _build_loader_kwargs(num_workers):
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    return loader_kwargs


def _prepare_train_batch(all_in, num_tasks):
    input, target = zip(*[all_in[k] for k in range(num_tasks)])

    if args.distributed:
        slice_pt = 0
        slice_idx = [0]
        for batch in input:
            slice_pt += batch.size(0)
            slice_idx.append(slice_pt)
        input = torch.cat(input, dim=0)
        target = torch.cat(target, dim=0)
        return input, target, slice_idx

    if num_tasks == 1:
        input = input[0]
        target = target[0]
        local_batch = input.size(0) // args.ngpu
        return input, target, [0, local_batch]

    slice_pt = 0
    slice_idx = [0]
    for batch in input:
        slice_pt += batch.size(0) // args.ngpu
        slice_idx.append(slice_pt)

    organized_input = []
    organized_target = []
    for ng in range(args.ngpu):
        for task_idx in range(num_tasks):
            batch_size = input[task_idx].size(0) // args.ngpu
            start = ng * batch_size
            end = (ng + 1) * batch_size
            organized_input.append(input[task_idx][start:end, ...])
            organized_target.append(target[task_idx][start:end, ...])
    input = torch.cat(organized_input, dim=0)
    target = torch.cat(organized_target, dim=0)
    return input, target, slice_idx


def _should_save_eval_features():
    return getattr(args.test, 'save_features', False)


def _evaluation_output_path(epoch, benchmark):
    if not _should_save_eval_features():
        return None
    return "{}/checkpoints/ckpt_epoch_{}_{}.bin".format(args.save_path, epoch, benchmark)

def main():
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    ## config
    global args
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.opts:
        for i in range(0, len(args.opts), 2):
            if i + 1 >= len(args.opts):
                log("Warning: odd number of options provided, ignoring the last one: {}".format(args.opts[i]))
                break
            key = args.opts[i]
            val = args.opts[i+1]
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = yaml.safe_load(val)

    def dict_to_argobj(d):
        obj = ArgObj()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(obj, k, dict_to_argobj(v))
            else:
                setattr(obj, k, v)
        return obj

    for k, v in config.items():
        if isinstance(v, dict):
            setattr(args, k, dict_to_argobj(v))
        else:
            setattr(args, k, v)
    args.ngpu = len(args.gpus.split(','))
    
    if args.demo:
        log("DEMO MODE ENABLED: Limiting training and evaluation for fast verification.")
        args.train.max_epoch = 1
        args.train.print_freq = 1
        args.test.benchmark = ['lfw']
        args.test.interval = 1
    
    # DDP Initialization
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        # Prioritize LOCAL_RANK environment variable set by torchrun
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
        log("Distributed training initialized: rank {}/{}".format(args.rank, args.world_size))

    ## asserts
    assert args.model.backbone in model_names, "available backbone names: {}".format(model_names)
    num_tasks = len(args.train.data_root)
    assert(num_tasks == len(args.train.loss_weight))
    assert(num_tasks == len(args.train.batch_size))
    assert(num_tasks == len(args.train.data_list))
    #assert(num_tasks == len(args.train.data_meta))
    if args.val.flag:
        assert(num_tasks == len(args.val.batch_size))
        assert(num_tasks == len(args.val.data_root))
        assert(num_tasks == len(args.val.data_list))
        #assert(num_tasks == len(args.val.data_meta))

    ## mkdir
    if not args.save_path:
        args.save_path = os.path.dirname(args.config)
    os.makedirs('{}/checkpoints'.format(args.save_path), exist_ok=True)
    os.makedirs('{}/logs'.format(args.save_path), exist_ok=True)
    os.makedirs('{}/events'.format(args.save_path), exist_ok=True)

    ## create dataset
    if not (args.extract or args.evaluate): # train + val
        # In DDP, batch_size is per GPU. In DP, it's global.
        # We'll stick to per-GPU batch size for DDP and multiply for DP if needed.
        if not args.distributed:
            for i in range(num_tasks):
                args.train.batch_size[i] *= args.ngpu

        #train_dataset = [FaceDataset(args, idx, 'train') for idx in range(num_tasks)]
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
        train_longest_size = max([int(np.ceil(len(td) / float(bs))) for td, bs in zip(train_dataset, args.train.batch_size)])
        train_sampler = [GivenSizeSampler(td, total_size=train_longest_size * bs, rand_seed=args.train.rand_seed) for td, bs in zip(train_dataset, args.train.batch_size)]
        train_loader = [DataLoader(
            train_dataset[k], batch_size=args.train.batch_size[k], shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler[k],
            **_build_loader_kwargs(args.workers)) for k in range(num_tasks)]
        assert(all([len(train_loader[k]) == len(train_loader[0]) for k in range(num_tasks)]))

        if args.val.flag:
            if not args.distributed:
                for i in range(num_tasks):
                    args.val.batch_size[i] *= args.ngpu
    
            #val_dataset = [FaceDataset(args, idx, 'val') for idx in range(num_tasks)]
            val_dataset = [FileListLabeledDataset(
                args.val.data_list[i], args.val.data_root[i],
                transforms.Compose([
                    transforms.Resize(args.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]),
                memcached=args.memcached,
                memcached_client=args.memcached_client) for i in range(num_tasks)]
            
            val_longest_size = max([int(np.ceil(len(vd) / float(bs))) for vd, bs in zip(val_dataset, args.val.batch_size)])
            val_sampler = [GivenSizeSampler(vd, total_size=val_longest_size * bs, sequential=True) for vd, bs in zip(val_dataset, args.val.batch_size)]
            val_loader = [DataLoader(
                val_dataset[k], batch_size=args.val.batch_size[k], shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=val_sampler[k],
                **_build_loader_kwargs(args.workers)) for k in range(num_tasks)]
            assert(all([len(val_loader[k]) == len(val_loader[0]) for k in range(num_tasks)]))

    if args.test.flag or args.evaluate: # online or offline evaluate
        if not args.distributed:
            args.test.batch_size *= args.ngpu
        test_dataset = []
        for tb in args.test.benchmark:
            if tb == 'megaface':
                test_dataset.append(FileListDataset(args.test.megaface_list,
                    args.test.megaface_root, transforms.Compose([
                    transforms.Resize(args.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])))
            else:
                test_dataset.append(BinDataset("{}/{}.bin".format(args.test.test_root, tb),
                    transforms.Compose([
                    transforms.Resize(args.model.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])))
        test_sampler = [GivenSizeSampler(td,
            total_size=int(np.ceil(len(td) / float(args.test.batch_size)) * args.test.batch_size),
            sequential=True, silent=True) for td in test_dataset]
        test_loader = [DataLoader(
            td, batch_size=args.test.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=ts,
            **_build_loader_kwargs(args.workers))
            for td, ts in zip(test_dataset, test_sampler)]

    if args.extract: # feature extraction
        if not args.distributed:
            args.extract_info.batch_size *= args.ngpu
#        extract_dataset = FaceDataset(args, 0, 'extract')
        extract_dataset = FileListDataset(
            args.extract_info.data_list, args.extract_info.data_root,
            transforms.Compose([
                transforms.Resize(args.model.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]),
            memcached=args.memcached,
            memcached_client=args.memcached_client)
        extract_sampler = GivenSizeSampler(
            extract_dataset, total_size=int(np.ceil(len(extract_dataset) / float(args.extract_info.batch_size)) * args.extract_info.batch_size), sequential=True)
        extract_loader = DataLoader(
            extract_dataset, batch_size=args.extract_info.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=extract_sampler,
            **_build_loader_kwargs(args.workers))


    ## create model
    log("Creating model on [{}] gpus: {}".format(args.ngpu, args.gpus))
    if args.evaluate or args.extract:
        args.num_classes = None

    # Parse knowledge distillation config
    kd_config = None
    if hasattr(args, 'knowledge_distillation'):
        kd_obj = args.knowledge_distillation
        kd_config = {
            'enabled': getattr(kd_obj, 'enabled', False),
            'teacher_backbone': getattr(kd_obj, 'teacher_backbone', ''),
            'teacher_checkpoint': getattr(kd_obj, 'teacher_checkpoint', ''),
            'teacher_feature_dim': getattr(kd_obj, 'teacher_feature_dim', args.model.feature_dim),
            'teacher_input_size': getattr(kd_obj, 'teacher_input_size', args.model.input_size),
            'alpha': getattr(kd_obj, 'alpha', 1.0),
            'beta': getattr(kd_obj, 'beta', 0.5),
            'temperature': getattr(kd_obj, 'temperature', 1.0),
            'loss_type': getattr(kd_obj, 'loss_type', 'cosine'),
        }
        if kd_config['enabled']:
            log("Knowledge Distillation ENABLED: teacher={}, alpha={}, beta={}, temperature={}, loss_type={}".format(
                kd_config['teacher_backbone'], kd_config['alpha'], kd_config['beta'], kd_config['temperature'], kd_config['loss_type']))

    model = models.MultiTaskWithLoss(
        backbone=args.model.backbone, num_classes=args.num_classes,
        feature_dim=args.model.feature_dim, spatial_size=args.model.input_size,
        arc_fc=args.model.arc_fc, feat_bn=args.model.feat_bn,
        loss_type=getattr(args.model, 'loss_type', 'crossentropy'),
        kd_config=kd_config)
    
    if args.distributed:
        model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = nn.DataParallel(model)
        model.cuda()
    
    cudnn.benchmark = True
    
    # Initialize AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    ## criterion and optimizer
    opt_name = getattr(args.train, 'optimizer', 'sgd').lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.train.base_lr,
                                    momentum=args.train.momentum,
                                    weight_decay=args.train.weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.train.base_lr,
                                     weight_decay=args.train.weight_decay)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.train.base_lr,
                                      weight_decay=args.train.weight_decay)
    else:
        raise ValueError("Unsupported optimizer: {}".format(opt_name))

    ## resume / load model
    start_epoch = 0
    count = [0]
    checkpoint = None
    if args.load_path:
        assert os.path.isfile(args.load_path), "File not exist: {}".format(args.load_path)
        if args.resume:
            checkpoint = load_state(args.load_path, model, optimizer, scaler)
            start_epoch = checkpoint['epoch']
            count[0] = checkpoint['count']
        else:
            load_state(args.load_path, model)

    ## offline evaluate
    if args.evaluate:
        for tb, tl, td in zip(args.test.benchmark, test_loader, test_dataset):
            evaluation(tl, model, num=len(td),
                       outfeat_fn="{}_{}.bin".format(args.load_path[:-8], tb),
                       benchmark=tb,
                       labels=getattr(td, 'lbs', None))
        return

    ## feature extraction
    if args.extract:
        extract(extract_loader, model, num=len(extract_dataset), output_file="{}_{}.bin".format(args.load_path[:-8], args.extract_info.data_name))
        return

    ## lr scheduler
    steps_per_epoch = len(train_loader[0])
    warmup_epochs = getattr(args.train, 'warmup_epochs', 5)
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_start_factor = getattr(args.train, 'warmup_start_factor', 0.1)
    
    total_steps = args.train.max_epoch * steps_per_epoch
    min_lr = getattr(args.train, 'min_lr', 0)

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, end_factor=1.0,
            total_iters=warmup_steps
        )
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_lr_scheduler],
            milestones=[warmup_steps], last_epoch=count[0]-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
            eta_min=min_lr,
            last_epoch=count[0]-1
        )

    if args.resume and checkpoint is not None and 'scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            log("=> loaded scheduler state from checkpoint")
        except Exception as e:
            log("=> Warning: failed to load scheduler state ({}). Resuming via last_epoch instead.".format(e))

    ## logger
    if args.rank == 0:
        logging.basicConfig(filename=os.path.join('{}/logs'.format(args.save_path), 'log-{}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}.txt'.format(
            datetime.today().year, datetime.today().month, datetime.today().day,
            datetime.today().hour, datetime.today().minute, datetime.today().second)),
            level=logging.INFO)
        tb_logger = SummaryWriter('{}/events'.format(args.save_path))
    else:
        tb_logger = None

    ## initial validate
    if args.val.flag:
        # torch.cuda.empty_cache()
        validate(val_loader, model, start_epoch, args.train.loss_weight, len(train_loader[0]), tb_logger, count)

    ## initial evaluate
    if args.test.flag and args.test.initial_test:
        # torch.cuda.empty_cache()
        log("*************** evaluation epoch [{}] ***************".format(start_epoch))
        for tb, tl, td in zip(args.test.benchmark, test_loader, test_dataset):
            res = evaluation(tl, model, num=len(td),
                             outfeat_fn=_evaluation_output_path(start_epoch, tb),
                             benchmark=tb,
                             labels=getattr(td, 'lbs', None))
            if tb_logger:
                tb_logger.add_scalar(tb, res, start_epoch)

    ## training loop
    for epoch in range(start_epoch, args.train.max_epoch):
        for ts in train_sampler:
            ts.set_epoch(epoch)
        # train for one epoch
        # train for one epoch
        train(train_loader, model, optimizer, epoch, args.train.loss_weight, tb_logger, count, scaler, lr_scheduler)
        # save checkpoint
        if args.rank == 0:
            save_state({
                'epoch': epoch + 1,
                'arch': args.model.backbone,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'count': count[0],
                'scaler': scaler.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, args.save_path + "/checkpoints/ckpt_epoch", epoch + 1, is_last=(epoch + 1 == args.train.max_epoch))

        # validate
        if args.val.flag:
            # torch.cuda.empty_cache()
            validate(val_loader, model, epoch, args.train.loss_weight, len(train_loader[0]), tb_logger, count)
        # online evaluate
        if args.test.flag and ((epoch + 1) % args.test.interval == 0 or epoch + 1 == args.train.max_epoch):
            # torch.cuda.empty_cache()
            log("*************** evaluation epoch [{}] ***************".format(epoch + 1))
            for tb, tl, td in zip(args.test.benchmark, test_loader, test_dataset):
                res = evaluation(tl, model, num=len(td),
                                 outfeat_fn=_evaluation_output_path(epoch + 1, tb),
                                 benchmark=tb,
                                 labels=getattr(td, 'lbs', None))
                if tb_logger:
                    tb_logger.add_scalar(tb, res, epoch + 1)



    if args.distributed:
        dist.destroy_process_group()


def train(train_loader, model, optimizer, epoch, loss_weight, tb_logger, count, scaler, lr_scheduler):
    num_tasks = len(train_loader)
    batch_time = AverageMeter(args.train.average_stats)
    data_time = AverageMeter(args.train.average_stats)
    losses = [AverageMeter(args.train.average_stats) for k in range(num_tasks)]

    # KD tracking
    kd_enabled = hasattr(args, 'knowledge_distillation') and getattr(args.knowledge_distillation, 'enabled', False)
    kd_alpha = getattr(args.knowledge_distillation, 'alpha', 1.0) if kd_enabled else 1.0
    kd_beta = getattr(args.knowledge_distillation, 'beta', 0.5) if kd_enabled else 0.0
    kd_losses = AverageMeter(args.train.average_stats) if kd_enabled else None

    # switch to train mode
    model.train()

    end = time.time()
    for i, all_in in enumerate(zip(*tuple(train_loader))):
        if args.demo and i >= 10:
            log("Demo mode: stopping epoch early at iteration {}".format(i))
            break
        input, target, slice_idx = _prepare_train_batch(all_in, num_tasks)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda"):
            # measure accuracy and record loss
            task_losses, kd_loss = model(input, target, slice_idx)
            
            task_loss_total = 0.
            for k in range(num_tasks):
                task_loss_total = task_loss_total + task_losses[k].mean() * loss_weight[k]

            # Combine task loss with KD loss
            if kd_loss is not None:
                loss_total = kd_alpha * task_loss_total + kd_beta * kd_loss
            else:
                loss_total = task_loss_total

        # scale loss and backprop
        scaler.scale(loss_total).backward()
        
        # unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        for k in range(num_tasks):
            if torch.__version__ >= '1.1.0':
                losses[k].update(task_losses[k].mean().item()) 
            else:
                losses[k].update(task_losses[k].mean().data[0])
        if kd_loss is not None:
            kd_losses.update(kd_loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # info
        if i % args.train.print_freq == 0:
            log('Progress: [{0}][{1}/{2}][{3}]    '
                  'Lr: {4:.2g}    '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})    '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                   epoch, i, len(train_loader[0]), count[0], 
                   optimizer.param_groups[0]['lr'],
                   batch_time=batch_time,
                   data_time=data_time))
            for k in range(num_tasks):
                log('Task: #{0}\t'
                      'LW: {1:.2g}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       k, loss_weight[k], loss=losses[k]))
            if kd_losses is not None:
                log('KD:\talpha: {0:.2g}\tbeta: {1:.2g}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                       kd_alpha, kd_beta, loss=kd_losses))

        # tensorboard logger
        if tb_logger:
            for k in range(num_tasks):
                tb_logger.add_scalar('train_loss_{}'.format(k), losses[k].val, count[0])
            if kd_losses is not None:
                tb_logger.add_scalar('train_kd_loss', kd_losses.val, count[0])
            tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], count[0])

        count[0] += 1

def validate(val_loader, model, epoch, loss_weight, train_len, tb_logger, count):
    log("Validation not fully implemented in this script. Skipping...")
    return
    num_tasks = len(val_loader)
    losses = [AverageMeter(args.val.average_stats) for k in range(num_tasks)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, all_in in enumerate(zip(*tuple(val_loader))):
            input, target = zip(*[all_in[k] for k in range(num_tasks)])

            slice_pt = 0
            slice_idx = [0]
            for l in [p.size(0) for p in input]:
                slice_pt += l
                slice_idx.append(slice_pt)

            input = torch.cat(tuple(input), dim=0).cuda()
            target = [tg.cuda() for tg in target]

            # measure accuracy and record loss
            loss = model(input, target, slice_idx)

        for k in range(num_tasks):
            if torch.__version__ >= '1.1.0':
                losses[k].update(loss[k].item())
            else:
                losses[k].update(loss[k].data[0])

    log('Test epoch #{}    Time {}'.format(epoch, time.time() - start))
    for k in range(num_tasks):
        log(' * Task: #{0}    Loss {loss.avg:.4f}'.format(k, loss=losses[k]))

    for k in range(num_tasks):
        tb_logger.add_scalar('val_loss_{}'.format(k), losses[k].val, count[0])

def extract(ext_loader, model, num, output_file, silent=False):
    batch_time = AverageMeter(9999999)
    data_time = AverageMeter(9999999)
    model.eval()
    features = []

    start = time.time()
    end = time.time()
    autocast_enabled = torch.cuda.is_available()
    with torch.inference_mode():
        for i, input in enumerate(ext_loader):
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                output = model(input, extract_mode=True)
            features.append(output.detach().float().cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.train.print_freq == 0 and not silent:
                log("Extracting: {0}/{1}\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        i, len(ext_loader), batch_time=batch_time, data_time=data_time))

    features = np.concatenate(features, axis=0)[:num, :]
    if output_file is not None:
        features.tofile(output_file)
    if not silent:
        log("Extracting Done. Total time: {}".format(time.time() - start))
    return features

def evaluation(test_loader, model, num, outfeat_fn, benchmark, labels=None):
    load_feat = False
    if outfeat_fn is None or not os.path.isfile(outfeat_fn) or not load_feat:
        features = extract(test_loader, model, num, None, silent=True)
        
        # Gather features from all ranks in DDP
        if args.distributed:
            local_features = torch.from_numpy(features).cuda()
            all_features = [torch.zeros_like(local_features) for _ in range(args.world_size)]
            dist.all_gather(all_features, local_features)
            
            # Efficiently interleave to restore original order (round-robin distribution in GivenSizeSampler)
            all_features_t = torch.stack(all_features, dim=0) # (world_size, local_num, feat_dim)
            all_features_t = all_features_t.transpose(0, 1) # (local_num, world_size, feat_dim)
            features = all_features_t.reshape(-1, local_features.size(1)).cpu().numpy()[:num, :]

        # Only rank 0 writes to file
        if args.rank == 0 and outfeat_fn is not None:
            features.tofile(outfeat_fn)
    else:
        if args.rank == 0:
            log("loading from: {}".format(outfeat_fn))
        features = np.fromfile(outfeat_fn, dtype=np.float32).reshape(-1, args.model.feature_dim)

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
            features, lbs, nrof_folds=args.test.nfolds, distance_metric=0)
    
        log(" * {}: accuracy: {:.4f}({:.4f})".format(benchmark, acc.mean(), acc.std()))
        return acc.mean()


#def evaluation_old(test_loader, model, num, outfeat_fn, benchmark):
#    load_feat = False
#    if not os.path.isfile(outfeat_fn) or not load_feat:
#        features = extract(test_loader, model, num, outfeat_fn)
#    else:
#        log("Loading features: {}".format(outfeat_fn))
#        features = np.fromfile(outfeat_fn, dtype=np.float32).reshape(-1, args.model.feature_dim)
#
#    if benchmark == "megaface":
#        r = test.test_megaface(features)
#        log(' * Megaface: 1e-6 [{}], 1e-5 [{}], 1e-4 [{}]'.format(r[-1], r[-2], r[-3]))
#        with open(outfeat_fn[:-4] + ".txt", 'w') as f:
#            f.write(' * Megaface: 1e-6 [{}], 1e-5 [{}], 1e-4 [{}]'.format(r[-1], r[-2], r[-3]))
#        return r[-1]
#    elif benchmark == "ijba":
#        r = test.test_ijba(features)
#        log(' * IJB-A: {} [{}], {} [{}], {} [{}]'.format(r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1]))
#        with open(outfeat_fn[:-4] + ".txt", 'w') as f:
#            f.write(' * IJB-A: {} [{}], {} [{}], {} [{}]'.format(r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1]))
#        return r[2][1]
#    elif benchmark == "lfw":
#        r = test.test_lfw(features)
#        log(' * LFW: mean: {} std: {}'.format(r[0], r[1]))
#        with open(outfeat_fn[:-4] + ".txt", 'w') as f:
#            f.write(' * LFW: mean: {} std: {}'.format(r[0], r[1]))
#        return r[0]


if __name__ == '__main__':
    main()
