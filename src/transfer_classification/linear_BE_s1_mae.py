# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import models.mae.util.misc as misc
from models.mae.util.pos_embed import interpolate_pos_embed
from models.mae.util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.mae.util.lars import LARS
from models.mae.util.crop import RandomResizedCrop
import models.mae.util.lr_decay as lrd

import models.mae.models_vit as models_vit

from models.mae.engine_finetune_BE import train_one_epoch, evaluate


from datasets.BigEarthNet.bigearthnet_dataset_official_lmdb_s1_float32 import LMDBDataset,random_subset
from cvtorchvision import cvtransforms
from sklearn.metrics import average_precision_score


def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    
    parser.add_argument("--is_slurm_job", action='store_true', help="slurm job")
    parser.add_argument("--train_frac", default=1.0, type=float, help="use a subset of labeled data")
    parser.add_argument("--fine_tune", action='store_true', help="fine tune or not")
    
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    train_transforms = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(224,scale=(0.8,1.0)),
            cvtransforms.RandomHorizontalFlip(),
            cvtransforms.ToTensor()])

    val_transforms = cvtransforms.Compose([
            cvtransforms.Resize(256),
            cvtransforms.CenterCrop(224),
            cvtransforms.ToTensor(),
            ])


    dataset_train = LMDBDataset(
        lmdb_file=os.path.join(args.data_path, 'BigEarthNet_LMDB_raw/train_B12_B2.lmdb'),
        transform=train_transforms,
        is_slurm_job=args.is_slurm_job,
        bands='B2'            
    )

    if not args.eval:
        dataset_val = LMDBDataset(
            lmdb_file=os.path.join(args.data_path, 'BigEarthNet_LMDB_raw/val_B12_B2.lmdb'),
            transform=val_transforms,
            is_slurm_job=args.is_slurm_job,
            bands='B2'            
        )

    else:
        dataset_val = LMDBDataset(
            lmdb_file=os.path.join(args.data_path, 'BigEarthNet_LMDB_raw/test_B12_B2.lmdb'),
            transform=val_transforms,
            is_slurm_job=args.is_slurm_job,
            bands='B2'            
        )           

    dataset_test = LMDBDataset(
        lmdb_file=os.path.join(args.data_path, 'BigEarthNet_LMDB_raw/test_B12_B2.lmdb'),
        #lmdb_file=os.path.join(args.data_path, 'seco_split/val_B12.lmdb'),
        transform=val_transforms,
        is_slurm_job=args.is_slurm_job,
        bands='B2'       
    )
    
    if args.train_frac is not None and args.train_frac<1:
        dataset_train = random_subset(dataset_train,args.train_frac,seed=42)        


    if True:  # args.distributed:
        num_tasks = args.world_size
        print(misc.get_world_size())
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        in_chans=2
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    if not args.fine_tune:
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * args.world_size
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #if not args.fine_tune:
    #optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.fine_tune:
        optimizer = torch.optim.SGD(model_without_ddp.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        ## build optimizer with layer-wise lr decay (lrd)
        #param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        #    no_weight_decay_list=model_without_ddp.no_weight_decay(),
        #    layer_decay=args.layer_decay
        #)
        #optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model_without_ddp.head.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)

                                
    print(optimizer)
    loss_scaler = NativeScaler()

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, criterion)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc_micro']:.1f}% {test_stats['acc_macro']:.1f}% {test_stats['f1_micro']:.2f}% {test_stats['f1_macro']:.2f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_acc_epoch = 0
    max_acc_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch%20==0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch==args.start_epoch or epoch%20==0 or (epoch + 1 == args.epochs):
            test_stats = evaluate(data_loader_val, model, device, criterion)
            print(f"Accuracy of the network on the {len(dataset_val)} val images: {test_stats['acc_micro']:.2f}% {test_stats['acc_macro']:.2f}% {test_stats['f1_micro']:.2f}% {test_stats['f1_macro']:.2f}%")
            
            if test_stats["acc_micro"] > max_accuracy:
                max_acc_epoch = epoch
                max_accuracy = test_stats["acc_micro"]
                ## testset
                test_stats1 = evaluate(data_loader_test, model, device, criterion)
                print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats1['acc_micro']:.2f}% {test_stats1['acc_macro']:.2f}% {test_stats1['f1_micro']:.2f}% {test_stats1['f1_macro']:.2f}%")
                max_acc_test = test_stats1['acc_micro']
            #max_accuracy = max(max_accuracy, test_stats["acc_micro"])
            print(f'Max val accuracy: {max_accuracy:.2f}% epoch: {max_acc_epoch:d}% test acc: {max_acc_test:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc_micro', test_stats['acc_micro'], epoch)
            log_writer.add_scalar('perf/test_acc_macro', test_stats['acc_macro'], epoch)
            log_writer.add_scalar('perf/test_f1_micro', test_stats['f1_micro'], epoch)
            log_writer.add_scalar('perf/test_f1_macro', test_stats['f1_macro'], epoch)            
            
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
