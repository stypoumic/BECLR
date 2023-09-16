from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.mask_loader import ImageFolderMask
from evaluate import evaluate_fewshot
from memory import NNmemoryBankModule
from models.beclr import ClusterLoss, BECLRLoss
from transform.build_transform import DataAugmentationBECLR
from utils import (LARS, AverageMeter, apply_mask_resnet, bool_flag,
                   build_fewshot_loader, build_student_teacher,
                   cancel_gradients_last_layer, cosine_scheduler,
                   fix_random_seeds, get_params_groups, get_world_size,
                   grad_logger, init_distributed_mode, load_student_teacher,
                   save_student_teacher, build_train_loader)

# torch.cuda.empty_cache()


def args_parser():
    parser = argparse.ArgumentParser(
        'BECLR training arguments', add_help=False)

    parser.add_argument('--save_path', type=str,
                        default=None, help='path for saving checkpoints')
    parser.add_argument('--log_path', type=str,
                        default=None, help='path for tensorboard logger')
    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset root')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet', 'miniImageNet',
                                 'CIFAR-FS', 'FC100'],
                        help='choice of dataset for pre-training')
    parser.add_argument('--print_freq', type=int,
                        default=120, help='print frequency')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--ckpt_freq', type=int,
                        default=10, help='checkpoint save frequency')
    parser.add_argument('--ckpt_path', type=str,
                        default=None, help='path to model checkpoint')
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # model settings
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet10', 'resnet18',
                                 'resnet34', 'resnet50'],
                        help='Choice of backbone network for the encoder')
    parser.add_argument('--size', type=int, default=224,
                        help='input image size')
    parser.add_argument('--enhance_batch', default=False, type=bool_flag,
                        help="Whether to artificially enhance the batch size")
    parser.add_argument('--topk', default=10, type=int,
                        help='Number of topk NN to extract, when enhancing the \
                        batch size.')
    parser.add_argument('--use_cluster_select', default=True, type=bool_flag,
                        help="Whether to use memory clusters for selecting NN")
    parser.add_argument('--use_single_memory', type=bool_flag, default=False,
                        help="""Whether to use separate memories for student & \
                        teacher""")
    parser.add_argument('--out_dim', default=512, type=int,
                        help="""Dimensionality of output.""")
    parser.add_argument('--momentum_teacher', default=0.9, type=float,
                        help="""Base EMA parameter for teacher update. The value
                        is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs during which we keep the output\
                        layer fixed. Typically doing so during the first epoch \
                        helps training. """)

    # contrastive loss settings
    parser.add_argument('--uniformity_config', type=str, default='SS',
                        choices=['ST', 'SS', 'TT'],
                        help='Choice of unifmormity configurations for view 1\
                        and view 2(SS: both views from student, ST: one view\
                        from student & the other from teacher, TT: both views\
                        from teacher)')
    parser.add_argument('--temp', type=float, default=2.0,
                        help='temperature for loss function')
    parser.add_argument('--lamb_neg', type=float, default=0.1,
                        help='lambda for uniformity loss')
    parser.add_argument('--use_memory_in_loss', default=False, type=bool_flag,
                        help="Whether to use memory in uniformity loss")
    parser.add_argument('--pos_threshold', default=0.8, type=float,
                        help="""When the cosine similarity of two embeddings is \
                        above this threshold, they are treated as positives, \
                        and masked out from the uniformity loss""")
    parser.add_argument('--use_only_batch_neg', default=False, type=bool_flag,
                        help="use only original batch in uniformtiy")

    # memory settings
    parser.add_argument("--memory_scale", default=20, type=int,
                        help="memory size compared to number of clusters, i.e.:\
                        memory_size = memory_scale * num_clusters")
    parser.add_argument('--num_clusters', type=int,
                        default=100, help='number of memory clusters')
    parser.add_argument('--cluster_algo', type=str, default='kmeans',
                        choices=['kmeans'], help='Choice of clustering algorithm\
                        for initializing the memory clusters')
    parser.add_argument('--recluster', default=True, type=bool_flag,
                        help="""Wether to occasionally recluster the memory \
                        embeddings all together""")
    parser.add_argument('--cluster_freq', type=int,
                        default=60, help='memory reclustering frequency')
    parser.add_argument('--memory_start_epoch', default=50, type=int,
                        help=' Epoch after which enhance_batch is \
                        activated.')
    parser.add_argument('--memory_momentum', default=0.0, type=float,
                        help="""the momentum value for updating the cluster \
                        means in the memory""")
    parser.add_argument('--memory_dist_metric', type=str, default='euclidean',
                        choices=['cosine', 'euclidean'], help='Choice of \
                        distance metric for the OT cost matrix in the memory')
    parser.add_argument("--sinkhorn_iterations", default=10, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp \
                        algorithm")
    parser.add_argument("--visual_freq", default=30, type=int,
                        help="memory embeddings visualization frequency")

    # masking settings
    parser.add_argument('--patch_size', type=int, default=16,
                        help='size of input square patches for masking in \
                        pixels, default 16 (for 16x16 patches)')
    parser.add_argument('--mask_ratio', default=0.0, type=float, nargs='+',
                        help="""Ratio of masked-out patches. If a list of ratio\
                        is specified, one of them will be randomly choosed for\
                        each image.""")
    parser.add_argument('--mask_ratio_var', default=0, type=float, nargs='+',
                        help="""Variance of partial masking ratio. Length \
                        should be indentical to the length of mask_ratio. \
                        0 for disabling. """)
    parser.add_argument('--mask_shape', default='block',
                        type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--mask_start_epoch', default=0, type=int,
                        help="""Start epoch to perform masked image prediction.""")

    # optimization settings
    parser.add_argument('--use_fp16', type=bool_flag, default=True,
                        help="""Whether or not to use half precision for \
                        training. Improves training time and memory \
                        requirements, but can provoke instability and slight \
                        decay of performance.""")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer', choices=['adamw', 'lars', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.25,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=1.0e-04, help='weight decay')
    parser.add_argument('--min_lr', type=float,
                        default=1.0e-06, help='final learning rate')
    parser.add_argument('--weight_decay_end', type=float,
                        default=0.0001, help='final weight decay')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='number of warmup epochs')

    # clustering loss settings (Optional)
    parser.add_argument('--use_clustering_loss', default=False, type=bool_flag,
                        help="Whether to use clustering loss")
    parser.add_argument('--w_clustering', type=float,
                        default=0.2, help='weight of clustering loss')
    parser.add_argument('--cls_use_enhanced_batch', default=True, type=bool_flag,
                        help="Whether to use the enhanced batch for the \
                        clustering loss")
    parser.add_argument('--cls_use_both_views', default=True, type=bool_flag,
                        help="Whether to enforce consistency for both views in \
                        clustering loss")
    parser.add_argument('--cluster_loss_start_epoch', default=100, type=int,
                        help=' Epoch after which the clustering loss is \
                        activated.')

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5,
                        help='number of classes per episode')
    parser.add_argument('--n_query', type=int, default=15,
                        help='number of queries per episode')
    parser.add_argument('--n_test_task', type=int,
                        default=600, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--eval_freq', type=int,
                        default=50, help='evaluation frequency')

    # parallelization settings
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; see \
                        https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int,
                        help="""number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int,
                        help="""rank of this process: it is set automatically \
                        and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    parser.add_argument('--eucl_norm', default=True, type=bool_flag,
                        help="Whether normalize before applying eucl distance")

    parser.add_argument('--use_nnclr', default=False, type=bool_flag,
                        help="Whether to use the memory of nnclr")

    return parser


def train_beclr(args: dict):
    """
    Performs the self-supervised pre-training stage of the network.

    Arguments:
        - args (dict): parsed keyword arguments for training.
    """
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    # build test data loader
    test_loader = build_fewshot_loader(args, 'test')

    # build data augmentationss
    transform = DataAugmentationBECLR(args)

    if args.dataset in ["FC100", "CIFAR-FS"]:
        data_loader = build_train_loader(args, transform)
    else:
        if args.dataset == "miniImageNet":
            data_path = Path(args.data_path) / Path("miniimagenet_train")
        elif args.dataset == "tieredImageNet":
            data_path = Path(args.data_path) / Path("tieredimagenet_train")

        pred_size = args.patch_size
        # build training dataset with patch-level masking
        dataset = ImageFolderMask(
            data_path,
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.mask_ratio,
            pred_ratio_var=args.mask_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.mask_shape,
            pred_start_epoch=args.mask_start_epoch)

        # build train data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building model ... ============
    student, teacher = build_student_teacher(args)

    # ============ preparing loss ... ============
    beclr_loss = BECLRLoss(args, lamb_neg=args.lamb_neg,
                           temp=args.temp).cuda()
    cluster_loss = ClusterLoss(dist_metric=args.memory_dist_metric).cuda()

    # ============ preparing memory queue ... ============
    memory_size = (args.memory_scale * args.num_clusters //
                   (args.batch_size * 2) + 1) * args.batch_size * 2 + 1
    print("Memory Size: {} \n".format(memory_size))

    if args.use_single_memory:
        teacher_nn_replacer = None
        student_nn_replacer = NNmemoryBankModule(
            size=memory_size, origin="student")
        student_f_nn_replacer = NNmemoryBankModule(
            size=memory_size, origin="student_f")
    else:
        teacher_nn_replacer = NNmemoryBankModule(
            size=memory_size, origin="teacher")
        student_nn_replacer = NNmemoryBankModule(
            size=memory_size, origin="student")
        student_f_nn_replacer = NNmemoryBankModule(
            size=memory_size, origin="student_f")

    # ============ preparing logger ... ============

    # local_runs = os.path.join("runs", "9_{}_B-{}_L-{}_M-{}_D-{}_E-{}_D_{}_MP_{}_SE{}_top{}_CL{}-{}_W{}_{}_CL{}-{}-{}-{}".format(
    #     args.dataset, args.backbone, args.lr, args.mask_ratio[0], args.out_dim,
    #     args.momentum_teacher, args.dist, args.use_fp16, args.memory_start_epoch,
    #     args.topk, args.num_clusters, args.memory_scale, args.lamb_neg, args.uniformity_config,
    #     args.use_clustering_loss, args.cls_use_enhanced_batch, args.cls_use_both_views, args.seed))
    if args.log_path == None:
        local_runs = Path(args.save_path) / Path("logs")
    else:
        local_runs = Path(args.log_path)
    print("Log Path: {}".format(local_runs))
    print("Checkpoint Save Path: {} \n".format(args.save_path))

    # initialize tensorboard logger
    writer = SummaryWriter(log_dir=local_runs)

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        # to use with convnet and large batches
        optimizer = LARS(params_groups)
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        args.lr * (args.batch_size * get_world_size()
                   ) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1 with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                         args.epochs, len(data_loader))

    print(f"Loss, optimizer and schedulers ready.")

    start_epoch = 0
    batch_size = args.batch_size
    # ============ Load checkpoint & Memory State ... ============
    if args.ckpt_path is not None:
        student, teacher, optimizer, fp16_scaler, start_epoch, loss, batch_size = load_student_teacher(
            student, teacher, args.ckpt_path, teacher_nn_replacer,
            student_nn_replacer, student_f_nn_replacer, optimizer=optimizer,
            fp16_scaler=fp16_scaler)

    start_time = time.time()
    print("Starting BECLR training!")
    for epoch in tqdm(range(start_epoch, args.epochs)):
        time1 = time.time()
        # data_loader.sampler.set_epoch(epoch)
        if args.dataset not in ["FC100", "CIFAR-FS"]:
            data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of BECLR ... ============
        loss = train_one_epoch(data_loader, student, teacher, optimizer,
                               fp16_scaler, epoch, lr_schedule, wd_schedule,
                               momentum_schedule, writer, beclr_loss, args,
                               teacher_nn_replacer, student_nn_replacer,
                               student_f_nn_replacer, cluster_loss)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch+1, time2 - time1))

        # ============ Save checkpoint & Memory State ... ============
        if args.save_path is not None and (epoch+1) % args.ckpt_freq == 0:
            fp16 = fp16_scaler.state_dict() if args.use_fp16 else None
            save_file = os.path.join(
                args.save_path, 'epoch_{}.pth'.format(epoch + 1))
            save_student_teacher(args, student, teacher, epoch + 1, loss,
                                 optimizer, batch_size, save_file,
                                 teacher_nn_replacer, student_nn_replacer,
                                 student_f_nn_replacer, fp16_scaler=fp16)

        # evaluate test performance every args.eval_freq epochs during training
        if (epoch) % args.eval_freq == 0 and epoch > 0:
            student.module.encoder.masked_im_modeling = False
            results = evaluate_fewshot(args, student.module.encoder,
                                       test_loader, n_way=args.n_way,
                                       n_shots=[1, 5], n_query=args.n_query,
                                       classifier='LR')
            student.module.encoder.masked_im_modeling = True
            # log accuracy and confidence intervals
            writer.add_scalar("1-Shot Accuracy", results[0][0], epoch+1)
            writer.add_scalar("5-Shot Accuracy", results[1][0], epoch+1)
            writer.add_scalar("1-Shot C95", results[0][1], epoch+1)
            writer.add_scalar("5-Shot C95", results[1][1], epoch+1)
        writer.flush()

    # ============ Evaluate Few Shot Test performance ... ============
    student.module.encoder.masked_im_modeling = False
    evaluate_fewshot(args, student.module.encoder,
                     test_loader, n_way=args.n_way, n_shots=[1, 5],
                     n_query=args.n_query, classifier='LR')
    student.module.encoder.masked_im_modeling = True

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


def train_one_epoch(train_loader: torch.utils.data.DataLoader,
                    student: nn.Module,
                    teacher: nn.Module,
                    optimizer: nn.Module,
                    fp16_scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    lr_schedule: np.array,
                    wd_schedule: np.array,
                    momentum_schedule: np.array,
                    writer: SummaryWriter,
                    beclr_loss: nn.Module,
                    args: dict,
                    teacher_nn_replacer: NNmemoryBankModule,
                    student_nn_replacer: NNmemoryBankModule,
                    student_f_nn_replacer: NNmemoryBankModule = None,
                    cluster_loss: nn.Module = None):
    """
    Performs one epoch of the self-supervised pre-training stage of the network.

    Arguments:
        - train_loader (torch.utils.data.DataLoader): train dataloader
        - student (nn.module): student network
        - teacher (nn.module): teacher network
        - optimizer (nn.module): optimizer module
        - fp16_scaler (torch.cuda.amp.GradScaler): half-precision module
        - epoch (int): current training epoch
        - lr_schedule (np.array): learning rate cosine schedule
        - wd_schedule (np.array): weight decay cosine schedule
        - momentum_schedule (np.array): teacher momentum cosine schedule
        - writer (SummaryWriter): TensorBoard SummaryWritter
        - beclr_loss (nn.module): contrastive loss module
        - args (dict): parsed keyword training arguments
        - teacher_nn_replacer: teacher memory queue 
        - student_nn_replacer: student memory queue 
        - student_f_nn_replacer: student projections memory queue (Optional)
        - cluster_loss: clustering loss module (Optional)
    """
    student.train()

    # initialize logging metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hist = AverageMeter()
    loss_pos_hist = AverageMeter()
    loss_neg_hist = AverageMeter()
    std_hist = AverageMeter()
    loss_cluster_hist = AverageMeter()

    end = time.time()

    # for it, (images, _, masks) in enumerate(tqdm(train_loader)):
    # for it, (images, _) in enumerate(tqdm(train_loader)):
    for it, data in enumerate(tqdm(train_loader)):
        images = data[0]
        data_time.update(time.time() - end)
        bsz = images[0].shape[0]

        if bsz != args.batch_size:
            continue

        # # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.module.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.module.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
        # get student & teacher parameters
        params_q = [param_q for name_q, param_q in zip(
            names_q, params_q) if name_q in names_common]
        params_k = [param_k for name_k, param_k in zip(
            names_k, params_k) if name_k in names_common]

        # update weight decay and learning rate according to their schedule
        global_it = len(train_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if "resnet" in args.backbone:
                param_group["weight_decay"] = wd_schedule[global_it]
            else:
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[global_it]

        # move images to gpu
        images = torch.cat([images[0], images[1]],
                           dim=0).cuda(non_blocking=True)
        # Add zero masking on the teacher branch
        if args.mask_ratio[0] > 0.0 and args.dataset not in ["FC100", "CIFAR-FS"]:
            masks = data[-1]
            masks = torch.cat([masks[0], masks[1]],
                              dim=0).cuda(non_blocking=True)
            masked_images = apply_mask_resnet(
                images, masks, args.patch_size)
        else:
            masked_images = images

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # pass images from student/teacher encoders
            p, z_student = student(masked_images)
            z_teacher = teacher(images)

            # concat the features of top-k neighbors for both student &
            # teacher if batch size increase is activated
            if args.enhance_batch:
                if args.use_single_memory:
                    z_teacher = student_f_nn_replacer.get_top_kNN(
                        z_teacher.detach(), epoch, args, k=args.topk, update=True)
                    p = student_nn_replacer.get_top_kNN(
                        p, epoch, args, k=args.topk, update=True)
                    z_student = student_f_nn_replacer.get_top_kNN(
                        z_student, epoch, args, k=args.topk, update=True)
                else:
                    z_teacher = teacher_nn_replacer.get_top_kNN(
                        z_teacher.detach(), epoch, args, k=args.topk, update=True)
                    p = student_nn_replacer.get_top_kNN(
                        p, epoch, args, k=args.topk, update=True)
                    z_student = student_f_nn_replacer.get_top_kNN(
                        z_student, epoch, args, k=args.topk, update=True)
            elif args.use_nnclr:
                z_teacher = teacher_nn_replacer.get_NN(
                    z_teacher.detach(), epoch, args, update=True)

            # calculate contrastive loss
            loss_state = beclr_loss(
                z_teacher, p, z_student, args, epoch=epoch,
                memory=student_nn_replacer.bank.cuda())

            # calculate clustering loss (if enabled)
            if args.use_clustering_loss and epoch > args.cluster_loss_start_epoch \
                    and teacher_nn_replacer.start_clustering:
                cl_loss = cluster_loss(
                    args, p, z_teacher, teacher_nn_replacer)
            else:
                cl_loss = 0

        loss = loss_state['loss'] + args.w_clustering * cl_loss

        # update student weights through backpropagation
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # update teacher weights through EMA
        with torch.no_grad():
            m = momentum_schedule[global_it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        loss_hist.update(loss.item(), bsz)
        loss_cluster_hist.update(cl_loss, bsz)
        loss_pos_hist.update(loss_state["loss_pos"].item(), bsz)
        loss_neg_hist.update(loss_state["loss_neg"].item(), bsz)
        std_hist.update(loss_state["std"].item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_pos {lossp.val:.3f} ({lossp.avg:.3f})\t'
                  'loss_neg {lossn.val:.3f} ({lossn.avg:.3f})\t'
                  'loss_cluster {losscluster.val:.3f} ({losscluster.avg:.3f})\t'
                  'std {std.val:.3f} ({std.avg:.3f})'.format(
                      epoch + 1, global_it + 1 - epoch * len(train_loader), len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=loss_hist, lossp=loss_pos_hist,
                      lossn=loss_neg_hist, losscluster=loss_cluster_hist, std=std_hist))
            sys.stdout.flush()

    # log weight gradients
    grad_stats = grad_logger(student.named_parameters())

    _new_lr = lr_schedule[global_it]
    _new_wd = wd_schedule[global_it]

    writer.add_scalar("Loss", loss_hist.avg, epoch+1)
    writer.add_scalar("Alignment Loss", loss_pos_hist.avg, epoch+1)
    writer.add_scalar("Uniformity Loss", loss_neg_hist.avg, epoch+1)
    writer.add_scalar("Standard Deviation", std_hist.avg, epoch+1)
    writer.add_scalar("Batch Time", batch_time.avg, epoch+1)
    writer.add_scalar("Data Time", data_time.avg, epoch+1)
    writer.add_scalar("Learning Rate", _new_lr, epoch+1)
    writer.add_scalar("Weight Decay", _new_wd, epoch+1)
    writer.add_scalar("Weight Gradient Average", grad_stats.avg, epoch+1)
    writer.add_scalar("Cluster Loss", loss_cluster_hist.avg, epoch+1)
    writer.flush()

    return loss_hist.avg


if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser(
        'BECLR training arguments', parents=[args_parser()])
    args = parser.parse_args()

    args.split_path = (Path(__file__).parent).joinpath('split')

    # initialize distributed parallel training & fix random seed
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    train_beclr(args)
