from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter
from models.msiam import MSiamLoss, swav_loss
from utils import get_params_groups, get_world_size, clip_gradients, \
    cancel_gradients_last_layer, build_fewshot_loader, load_distil_model, build_student_teacher
from utils import AverageMeter, grad_logger, apply_mask_resnet, bool_flag, LARS, cosine_scheduler, save_student_teacher, load_student_teacher, init_distributed_mode
from memory import NNmemoryBankModule
from transform.build_transform import DataAugmentationMSiam
from evaluate import evaluate_fewshot
from dataset.mask_loader import ImageFolderMask
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.backends.cudnn as cudnn

import os
import sys
import argparse
import datetime
import time

import torch
import torch.nn as nn
torch.cuda.empty_cache()


def args_parser():
    parser = argparse.ArgumentParser(
        'MSiam training arguments', add_help=False)

    parser.add_argument('--save_path', type=str,
                        default=None, help='path for saving checkpoints')
    parser.add_argument('--data_path', type=str,
                        default=None, help='path to dataset')
    parser.add_argument('--eval_path', type=str,
                        default=None, help='path to tested model')
    parser.add_argument('--teacher_path', type=str,
                        default=None, help='path to teacher model')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        choices=['tieredImageNet', 'miniImageNet'], help='dataset')
    parser.add_argument('--print_freq', type=int,
                        default=40, help='print frequency')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--ckpt_freq', type=int,
                        default=10, help='checkpoint frequency')
    parser.add_argument('--ckpt_path', type=str,
                        default=None, help='path to model checkpoint')
    parser.add_argument('--use_feature_align', default=False, type=bool_flag,
                        help="Whether to use feature alignment")
    parser.add_argument('--use_feature_align_teacher', default=False, type=bool_flag,
                        help="Whether to use feature alignment on the teacher features")
    parser.add_argument('--w_ori', type=float,
                        default=1.0, help='weight of original features (in case of refinement)')

    # clustering
    parser.add_argument('--use_clustering', default=False, type=bool_flag,
                        help="Whether to use online clustering")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--nmb_prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--world_size", default=2, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")

    # memory
    parser.add_argument('--topk', default=5, type=int,
                        help='Number of topk NN to extract, when enhancing the batch size (Default 5).')
    parser.add_argument('--enhance_batch', default=False, type=bool_flag,
                        help="Whether to artificially enhance the batch size")
    parser.add_argument('--use_nnclr', default=False, type=bool_flag,
                        help="Whether to use nnclr")
    parser.add_argument('--memory_start_epoch', default=50, type=int,
                        help=' Epoch after which NNCLR, or enhance_batch is activated (Default: 50).')
    parser.add_argument('--use_memory_in_loss', default=False, type=bool_flag,
                        help="Whether to use memory in uniformity loss")

    ######
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=512, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=False, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
                        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
                        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=False, type=bool_flag,
                        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--use_patches', default=False, type=bool_flag,
                        help="Whether to use patch-level loss")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    during which we keep the output layer fixed. Typically doing so during
    the first epoch helps training. Try increasing this value if the loss does not decrease.""")

    # Model settings
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet10', 'resnet18', 'resnet34', 'resnet50', 'deit_tiny'], help='Choice of backbone network for the encoder')
    parser.add_argument('--size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='size of input square patches for masking in pixels, default 16 (for 16x16 patches)')

    parser.add_argument('--mask_ratio', default=0.0, type=float, nargs='+', help="""Ratio of masked-out patches.
        If a list of ratio is specified, one of them will be randomly choosed for each image.""")
    parser.add_argument('--mask_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--mask_shape', default='block',
                        type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--mask_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help="""Drop path rate for student network.""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # optimization settings
    parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimizer', choices=['adamw', 'lars', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.04, help='weight decay')
    parser.add_argument('--min_lr', type=float,
                        default=1.0e-06, help='final learning rate')
    parser.add_argument('--weight_decay_end', type=float,
                        default=0.4, help='final weight decay')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of training epochs')
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help='value to clip gradients')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='number of warmup epochs')

    # self-supervision settings
    parser.add_argument('--temp', type=float, default=2.0,
                        help='temperature for loss function')
    parser.add_argument('--lamb_neg', type=float, default=0.1,
                        help='lambda for uniform loss')
    parser.add_argument('--lamb_patch', type=float, default=0.1,
                        help='lambda for patch loss')

    # few-shot evaluation settings
    parser.add_argument('--n_way', type=int, default=5, help='n_way')
    parser.add_argument('--n_query', type=int, default=15, help='n_query')
    parser.add_argument('--n_test_task', type=int,
                        default=400, help='total test few-shot episodes')
    parser.add_argument('--test_batch_size', type=int,
                        default=5, help='episode_batch_size')
    parser.add_argument('--eval_freq', type=int,
                        default=50, help='evaluation frequency')

    return parser


def train_msiam(args):
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    # build test data loader
    test_loader = build_fewshot_loader(args, 'test')

    transform = DataAugmentationMSiam(args)

    data_path = Path(args.data_path) / Path("miniimagenet_train")

    pred_size = args.patch_size
    dataset = ImageFolderMask(
        data_path,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.mask_ratio,
        pred_ratio_var=args.mask_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.mask_shape,
        pred_start_epoch=args.mask_start_epoch)
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    # sampler=sampler,
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
    # model = build_model(args)
    student, teacher = build_student_teacher(args)

    distil_model = load_distil_model(args) if args.dist else None

    # ============ preparing loss ... ============
    msiam_loss = MSiamLoss(args, lamb_neg=args.lamb_neg,
                           lamb_patch=args.lamb_patch,
                           temp=args.temp).cuda()

    # ============ preparing memory queue ... ============
    teacher_nn_replacer = NNmemoryBankModule(size=2 ** 16)
    student_nn_replacer = NNmemoryBankModule(size=2 ** 16)

    if args.use_nnclr:
        suffix = "NNCLR"
    elif args.enhance_batch:
        suffix = "BI"
    local_runs = os.path.join("runs", "B-{}_O-{}_L-{}_M-{}_D-{}_E-{}_D_{}_MP_{}_SE{}_{}_top{}_UM_{}".format(
        args.backbone, args.optimizer, args.lr, args.mask_ratio[0], args.out_dim,
        args.momentum_teacher, args.dist, args.use_fp16, args.memory_start_epoch,
        suffix, args.topk, args.use_memory_in_loss))
    print("Log Path: {}".format(local_runs))
    print("Checkpoint Save Path: {} \n".format(args.save_path))
    writer = SummaryWriter(log_dir=local_runs)
    # writer = SummaryWriter()

    # ============ preparing optimizer ... ============
    # params_groups = model.parameters()
    # params_groups = student.parameters()
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
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                         args.epochs, len(data_loader))

    print(f"Loss, optimizer and schedulers ready.")

    start_epoch = 0
    batch_size = args.batch_size
    # ============ Load checkpoint ... ============
    if args.ckpt_path is not None:
        student, teacher, optimizer, fp16_scaler, start_epoch, loss, batch_size = load_student_teacher(
            student, teacher, ckpt_path=args.ckpt_path, optimizer=optimizer, fp16_scaler=fp16_scaler, )

    start_time = time.time()
    print("Starting MSiam training!")
    for epoch in tqdm(range(start_epoch, args.epochs)):
        time1 = time.time()
        # data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of MSiam ... ============
        loss = train_one_epoch(data_loader, student, teacher, optimizer, fp16_scaler, epoch,
                               lr_schedule, wd_schedule, momentum_schedule, writer, msiam_loss,
                               args, teacher_nn_replacer, student_nn_replacer, distil_model)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch+1, time2 - time1))

        if args.save_path is not None and (epoch+1) % args.ckpt_freq == 0:
            fp16 = fp16_scaler.state_dict() if args.use_fp16 else None
            save_file = os.path.join(
                args.save_path, 'epoch_{}.pth'.format(epoch + 1))
            save_student_teacher(student, teacher, epoch + 1, loss,
                                 optimizer, batch_size, save_file, fp16_scaler=fp16)

        # evaluate test performance every 50 epochs
        if (epoch) % args.eval_freq == 0 and epoch > 0:
            student.module.encoder.masked_im_modeling = False
            results = evaluate_fewshot(student.module.encoder, student.module.use_transformers, test_loader, n_way=args.n_way, n_shots=[
                1, 5], n_query=args.n_query, classifier='LR', power_norm=True)
            student.module.encoder.masked_im_modeling = True
            writer.add_scalar("1-Shot Accuracy", results[0][0], epoch+1)
            writer.add_scalar("5-Shot Accuracy", results[1][0], epoch+1)
            writer.add_scalar("1-Shot C95", results[0][1], epoch+1)
            writer.add_scalar("5-Shot C95", results[1][1], epoch+1)
        writer.flush()

    # ============ Evaluate Few Shot Test performance ... ============
    student.module.encoder.masked_im_modeling = False
    evaluate_fewshot(student.module.encoder, student.module.use_transformers, test_loader, n_way=args.n_way, n_shots=[
                     1, 5], n_query=args.n_query, classifier='LR', power_norm=True)
    student.module.encoder.masked_im_modeling = True

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


def train_one_epoch(train_loader, student, teacher, optimizer, fp16_scaler, epoch,
                    lr_schedule, wd_schedule, momentum_schedule, writer, msiam_loss,
                    args, teacher_nn_replacer, student_nn_replacer, distil_model=None):
    """one epoch training"""
    student.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_hist = AverageMeter()
    loss_pos_hist = AverageMeter()
    loss_neg_hist = AverageMeter()
    loss_patch_hist = AverageMeter()
    loss_dist_hist = AverageMeter()
    loss_pos_ref_hist = AverageMeter()
    std_hist = AverageMeter()

    end = time.time()

    ################################
    # every image in batch has seperate mask
    # mask 0 => all False
    for it, (images, labels, masks) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)
        bsz = images[0].shape[0]

        # # common params
        names_q, params_q, names_k, params_k = [], [], [], []
        for name_q, param_q in student.module.named_parameters():
            names_q.append(name_q)
            params_q.append(param_q)
        for name_k, param_k in teacher.module.named_parameters():
            names_k.append(name_k)
            params_k.append(param_k)
        names_common = list(set(names_q) & set(names_k))
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
        masks = torch.cat([masks[0], masks[1]],
                          dim=0).cuda(non_blocking=True)

        # Add zero masking on the anchor branch in case of ResNet backbone
        if args.mask_ratio[0] > 0.0 and "resnet" in args.backbone:
            masked_images = apply_mask_resnet(
                images, masks, args.patch_size)
        else:
            masked_images = images

        if distil_model is not None:
            with torch.no_grad():
                # could also give unmasked images as input to teacher
                z_dist, _ = distil_model(masked_images)
                z_dist = z_dist.detach()
        else:
            z_dist = None

        if args.use_clustering:
            # normalize the prototypes
            with torch.no_grad():
                w_student = student.module.prototypes.weight.data.clone()
                w_student = nn.functional.normalize(w_student, dim=1, p=2)
                student.module.prototypes.weight.copy_(w_student)

                w_teacher = teacher.module.prototypes.weight.data.clone()
                w_teacher = nn.functional.normalize(w_teacher, dim=1, p=2)
                teacher.module.prototypes.weight.copy_(w_teacher)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if 'deit' in args.backbone:
                pass
            else:
                p, p_refined, p_dist, s_prototypes = student(masked_images)
                z, z_refined, t_prototypes = teacher(images)
                # replace teacher features with NN if NNCLR is activated
                if args.use_nnclr:
                    z = teacher_nn_replacer(
                        z.detach(), epoch, args, k=args.topk, update=True)
                # concat the features of top-5 neighbors for both student &
                # teacher if batch size increase is activated
                if args.enhance_batch:
                    z = teacher_nn_replacer.get_top_kNN(
                        z.detach(), epoch, args, k=args.topk, update=True)
                    p = student_nn_replacer.get_top_kNN(
                        p, epoch, args, k=args.topk, update=True)

                if args.use_clustering:
                    cluster_loss = swav_loss(args, s_prototypes, t_prototypes, student,
                                             teacher, student_nn_replacer,
                                             teacher_nn_replacer)

                loss_state = msiam_loss(
                    z, p, args,
                    p_refined=p_refined,
                    z_refined=z_refined,
                    epoch=epoch,
                    p_dist=p_dist,
                    z_dist=z_dist,
                    w_ori=args.w_ori,
                    memory=teacher_nn_replacer.bank.cuda())

        loss = loss_state['loss']
        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad > 0.0:
                param_norms = clip_gradients(
                    student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad > 0.0:
                # unscale the gradients of optimizer's assigned params in-place
                fp16_scaler.unscale_(optimizer)
                param_norms = clip_gradients(student, args.clip_grad)
            cancel_gradients_last_layer(epoch, student,
                                        args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[global_it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        loss_hist.update(loss.item(), bsz)
        loss_pos_hist.update(loss_state["loss_pos"].item(), bsz)
        loss_neg_hist.update(loss_state["loss_neg"].item(), bsz)
        if args.dist:
            loss_dist_hist.update(loss_state["loss_dist"].item(), bsz)
        if args.use_feature_align:
            loss_pos_ref_hist.update(loss_state["loss_pos_ref"].item(), bsz)
        if args.use_patches:
            loss_patch_hist.update(loss_state["loss_patch"].item(), bsz)
        std_hist.update(loss_state["std"].item(), bsz)
        # add evaluation logging every 50 epochs 100 episodes
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_pos {lossp.val:.3f} ({lossp.avg:.3f})\t'
                  'loss_neg {lossn.val:.3f} ({lossn.avg:.3f})\t'
                  'loss_patch {losspa.val:.3f} ({losspa.avg:.3f})\t'
                  'loss_dist {lossdist.val:.3f} ({lossdist.avg:.3f})\t'
                  'loss_pos_ref {losspref.val:.3f} ({losspref.avg:.3f})\t'
                  'std {std.val:.3f} ({std.avg:.3f})'.format(
                      epoch + 1, global_it + 1 - epoch * len(train_loader), len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=loss_hist, lossp=loss_pos_hist,
                      lossn=loss_neg_hist, losspa=loss_patch_hist, lossdist=loss_dist_hist,
                      losspref=loss_pos_ref_hist, std=std_hist))
            sys.stdout.flush()

    # log weight gradients
    grad_stats = grad_logger(student.named_parameters())

    _new_lr = lr_schedule[global_it]
    _new_wd = wd_schedule[global_it]

    writer.add_scalar("Loss", loss_hist.avg, epoch+1)
    writer.add_scalar("Alignment Loss", loss_pos_hist.avg, epoch+1)
    writer.add_scalar("Uniformity Loss", loss_neg_hist.avg, epoch+1)
    writer.add_scalar("Patch Loss", loss_patch_hist.avg, epoch+1)
    writer.add_scalar("Self-Distillation Loss", loss_dist_hist.avg, epoch+1)
    writer.add_scalar("Refined Alignment Loss", loss_pos_ref_hist.avg, epoch+1)
    writer.add_scalar("Standard Deviation", std_hist.avg, epoch+1)
    writer.add_scalar("Batch Time", batch_time.avg, epoch+1)
    writer.add_scalar("Data Time", data_time.avg, epoch+1)
    writer.add_scalar("Learning Rate", _new_lr, epoch+1)
    writer.add_scalar("Weight Decay", _new_wd, epoch+1)
    writer.add_scalar("Weight Gradient Average", grad_stats.avg, epoch+1)
    writer.flush()

    return loss_hist.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MSiam training arguments', parents=[args_parser()])
    args = parser.parse_args()

    args.split_path = (Path(__file__).parent).joinpath('split')
    args.dist = args.teacher_path is not None
    ##
    # init_distributed_mode(args)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    train_msiam(args)
