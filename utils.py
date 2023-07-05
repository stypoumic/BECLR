import argparse
import os
import sys
from pathlib import Path

import learn2learn as l2l
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torchvision import transforms

from dataset.cub import CUBirds200
from dataset.miniImageNet import miniImageNet
from dataset.sampler import EpisodeSampler
from dataset.tieredImageNet import tieredImageNet
from models import resnet10, resnet18, resnet34, resnet50
from models.beclr import BECLR


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        ############### TO REMOVE #################

        # use gloo backend for local runs on windows
        dist.init_process_group(
            backend="gloo",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.gpu)
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.barrier()
        return
        ############### TO REMOVE #################

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        # args.world_size = int(os.environ["SLURM_NNODES"]) * int(
        #     os.environ["SLURM_TASKS_PER_NODE"][0]
        # )
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        # use gloo backend for local runs on windows
        dist.init_process_group(
            backend="gloo",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.gpu)
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.barrier()
        return
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    # print(args.rank)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}) (gpu {}): {}'.format(
        args.rank, args.gpu, args.dist_url), flush=True)
    dist.barrier()
    # dist.init_process_group(backend="nccl")
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_student_teacher(args):
    model_dict = {'resnet10': resnet10, 'resnet18': resnet18,
                  'resnet34': resnet34, 'resnet50': resnet50}

    student_encoder = model_dict[args.backbone]()
    teacher_encoder = model_dict[args.backbone]()
    embed_dim = student_encoder.out_dim

    student = BECLR(encoder=student_encoder, dim_in=embed_dim,
                    args=args, is_teacher=False)
    teacher = BECLR(encoder=teacher_encoder, dim_in=embed_dim,
                    args=args, is_teacher=True)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # teacher = nn.parallel.DistributedDataParallel(teacher,
    #                                               device_ids=[local_rank],
    #                                               output_device=local_rank)
    # student = nn.parallel.DistributedDataParallel(student,
    #                                               device_ids=[local_rank],
    #                                               output_device=local_rank)
    teacher = nn.parallel.DistributedDataParallel(teacher,
                                                  device_ids=[args.gpu])
    student = nn.parallel.DistributedDataParallel(student,
                                                  device_ids=[args.gpu])
    # teacher = torch.nn.DataParallel(teacher)
    # student = torch.nn.DataParallel(student)

    # teacher and student start with the same weights
    teacher.module.load_state_dict(
        student.module.state_dict(), strict=False)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(student)
    return student, teacher


def build_cub_fewshot_loader(args, n_shot=5, download=False, mode='test'):
    """ Generates tasks from the specified Dataset
    Arguments:- 
      root: root folder of Omniglot dataset
      image_transforms: transforms to be applied to images before loading in the dataloader
      target_transforms: transforms to be applied to target classes before loading in the dataloader
      task_transforms: specify n_ways, k_shots, q_queries, num_tasks to create and classes to sample 
                    tasks from (if Omniglot) or mode: train/valid/test split to load (if MiniImageNet)
                    or mode: train/validation/test split to load (if CIFARFS)"""

    image_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([224, 224])])
    n_ways = args.n_way
    k_shots = n_shot
    q_shots = args.n_query
    num_tasks = args.n_test_task
    root = args.data_path

    cub = CUBirds200(root, mode, transform=image_transforms,
                     target_transform=None, download=download)
    dataset = l2l.data.MetaDataset(cub)

    trans = [
        l2l.data.transforms.FusedNWaysKShots(dataset,
                                             n=n_ways,
                                             k=k_shots + q_shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset)
    ]
    tasks = l2l.data.TaskDataset(
        dataset, task_transforms=trans, num_tasks=num_tasks)

    return tasks


def build_fewshot_loader(args, mode='test'):

    assert mode in ['train', 'val', 'test']

    resize_dict = {160: 182, 224: 256, 288: 330, 320: 366, 384: 438}
    resize_size = resize_dict[args.size]
    print('Image Size: {}({})'.format(args.size, resize_size))

    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    print('test_transform: ', test_transform)

    if args.dataset == 'miniImageNet':
        test_dataset = miniImageNet(
            data_path=Path(args.data_path) / Path("miniimagenet"),
            split_path=args.split_path,
            partition=mode,
            transform=test_transform)
    elif args.dataset == 'tieredImageNet':
        test_dataset = tieredImageNet(
            data_path=Path(args.data_path) / Path("imagenet") / Path("train"),
            split_path=args.split_path,
            partition=mode,
            transform=test_transform)
    else:
        raise ValueError(args.dataset)

    test_sampler = EpisodeSampler(
        test_dataset.labels, args.n_test_task//args.test_batch_size, args.n_way, 5+args.n_query, args.test_batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_sampler, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.num_workers)

    return test_loader


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * \
        (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def save_student_teacher(args, student, teacher, epoch, loss, optimizer, batch_size,
                         save_file, teacher_memory, student_memory,
                         student_proj_memory, fp16_scaler=None):
    print('==> Saving... \n')
    if args.use_single_memory:
        teacher_bank = student_proj_memory.bank
        teacher_ptr = student_proj_memory.bank_ptr
        teacher_labels = student_proj_memory.labels
        teacher_centers = student_proj_memory.centers
    else:
        teacher_bank = teacher_memory.bank
        teacher_ptr = teacher_memory.bank_ptr
        teacher_labels = teacher_memory.labels
        teacher_centers = teacher_memory.centers
    save_state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'fp16_scaler': fp16_scaler,
        'teacher_memory': (teacher_bank, teacher_ptr, teacher_labels, teacher_centers),
        'student_memory': (student_memory.bank, student_memory.bank_ptr, student_memory.labels, student_memory.centers),
        'student_proj_memory': (student_proj_memory.bank, student_proj_memory.bank_ptr, student_proj_memory.labels, student_proj_memory.centers)
    }
    torch.save(save_state, save_file)
    del save_state


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def load_student_teacher(student, teacher, ckpt_path, teacher_memory=None,
                         student_memory=None, student_proj_memory=None,
                         optimizer=None, fp16_scaler=None):
    print('==> Loading... \n')
    # open checkpoint file

    checkpoint = torch.load(ckpt_path)
    # checkpoint = torch.load(
    #     ckpt_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    # )

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']

    student.load_state_dict(checkpoint['student'])
    teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

    teacher_memory.load_memory_bank(checkpoint['teacher_memory'])
    student_memory.load_memory_bank(checkpoint['student_memory'])
    student_proj_memory.load_memory_bank(checkpoint['student_proj_memory'])

    del checkpoint
    return student, teacher, optimizer, fp16_scaler, epoch, loss, batch_size


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for name, param in named_params:
        if (param.grad is not None) and not (name.endswith('.bias') or len(param.shape) == 1):
            grad_norm = float(torch.norm(param.grad.data))
            stats.update(grad_norm)
            if 'qkv' in name:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def apply_mask_resnet(images, mask, patch_size=16, patch_stride=16):

    # unfold images into patches of 16 x 16
    patches = images.unfold(2, patch_size, patch_stride).unfold(
        3, patch_size, patch_stride)
    unfold_shape = patches.size()
    mask = ~mask
    mask_ = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand(patches.size())

    # apply mask
    patches_masked = patches * mask_

    # reshape to BS, 3, 196, 16, 16
    patches = patches.reshape(
        unfold_shape[0], unfold_shape[1], -1, patch_size, patch_size)

    # Fold patches back to original images
    patches_masked = patches_masked.contiguous().view(-1, patch_size, patch_size)
    images_orig = patches_masked.view(unfold_shape)
    output_c = unfold_shape[1]
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[3] * unfold_shape[5]
    images_orig = images_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    images_orig = images_orig.view(
        images.size(0), output_c, output_h, output_w)

    images = images_orig

    return images
