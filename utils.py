import math
import numpy as np
import torch
import argparse
import torch.optim as optim
import torch.distributed as dist
from torchvision import transforms
from pathlib import Path
import copy
import os
import sys
from torch import nn

from dataset.miniImageNet import miniImageNet
from dataset.tieredImageNet import tieredImageNet
from dataset.sampler import EpisodeSampler

from models.msiam import MSiam
from models import resnet10, resnet18, resnet34, resnet50, vit_tiny

from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


@torch.no_grad()
def visualize_memory_embeddings(memory: torch.Tensor, labels: torch.Tensor,
                                num_clusters: int, save_path: str,
                                epoch: int, origin: str):
    # ----------------- 3D tSNE------------------------------
    tsne = TSNE(n_components=3, verbose=1)
    tsne_proj = tsne.fit_transform(memory)

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # only plot the first 25 clusters
    num_clusters = 25
    ax.set_prop_cycle(color=[cmap(1.*i/num_clusters)
                      for i in range(num_clusters)])

    for lab in range(num_clusters):
        indices = labels == lab
        indices = np.random.permutation(indices)
        # only plot 5 samples/class
        if len(indices < 5):
            ax.scatter(tsne_proj[indices, 0],
                       tsne_proj[indices, 1],
                       tsne_proj[indices, 2],
                       label=lab,
                       alpha=0.5)
        else:
            ax.scatter(tsne_proj[indices[0:5], 0],
                       tsne_proj[indices[0:5], 1],
                       tsne_proj[indices[0:5], 2],
                       label=lab,
                       alpha=0.5)
    # plt.show()
    save_path = Path(save_path) / Path(origin)
    save_path.mkdir(parents=True, exist_ok=True)
    # save 3d interactive graph
    pickle.dump(fig, open(save_path / Path("E_"+str(epoch)+".pickle"), 'wb'))

   # ----------------- 2D tSNE------------------------------
    tsne = TSNE(n_components=2, verbose=1)
    tsne_proj = tsne.fit_transform(memory)

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # only plot the first 25 clusters
    num_clusters = 25
    ax.set_prop_cycle(color=[cmap(1.*i/num_clusters)
                      for i in range(num_clusters)])

    for lab in range(num_clusters):
        indices = labels == lab
        indices = np.random.permutation(indices)
        # only plot 5 samples/class
        if len(indices < 5):
            ax.scatter(tsne_proj[indices, 0],
                       tsne_proj[indices, 1],
                       label=lab,
                       alpha=0.5)
        else:
            ax.scatter(tsne_proj[indices[0:5], 0],
                       tsne_proj[indices[0:5], 1],
                       label=lab,
                       alpha=0.5)
    # save 2d graph
    plt.savefig(save_path / Path("E_"+str(epoch)+".png"))


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)
    # nccl
    dist.init_process_group(
        backend="gloo",
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


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


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d,
                nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_distil_model(args):
    use_transformers = True if 'deit' in args.backbone else False

    checkpoint = torch.load(args.teacher_path)

    model_encoder = resnet50()
    embed_dim = model_encoder.out_dim
    args.dist = False
    model = MSiam(encoder=model_encoder, dim_in=embed_dim, args=args,
                  use_transformers=use_transformers, is_teacher=True)
    args.dist = True

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    msg = model.load_state_dict(checkpoint['student'])
    print(f'load teacher model from: {args.teacher_path}, {msg}')
    model.eval()
    return model


def build_student_teacher(args):
    model_dict = {'resnet10': resnet10, 'resnet18': resnet18,
                  'resnet34': resnet34, 'resnet50': resnet50,
                  'deit_tiny': vit_tiny}

    use_transformers = True if 'deit' in args.backbone else False

    if use_transformers:
        student_encoder = model_dict[args.backbone](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher_encoder = model_dict[args.backbone](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student_encoder.embed_dim
    else:
        student_encoder = model_dict[args.backbone]()
        teacher_encoder = model_dict[args.backbone]()
        embed_dim = student_encoder.out_dim

    student = MSiam(encoder=student_encoder, dim_in=embed_dim, args=args,
                    use_transformers=use_transformers, is_teacher=False)
    teacher = MSiam(encoder=teacher_encoder, dim_in=embed_dim, args=args,
                    use_transformers=use_transformers, is_teacher=True)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    # if has_batchnorms(student) and use_transformers:
    #     student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    #     teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # # we need DDP wrapper to have synchro batch norms working...
    # if use_transformers:
    #     student = nn.parallel.DistributedDataParallel(student)
    #     teacher = nn.parallel.DistributedDataParallel(teacher)
    # else:
    teacher = torch.nn.DataParallel(teacher)
    student = torch.nn.DataParallel(student)

    # teacher and student start with the same weights
    teacher.module.load_state_dict(
        student.module.state_dict(), strict=False)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(student)
    return student, teacher


def build_model(args):
    model_dict = {'resnet10': resnet10, 'resnet18': resnet18,
                  'resnet34': resnet34, 'resnet50': resnet50,
                  'deit_tiny': vit_tiny}

    use_transformers = True if 'deit' in args.backbone else False

    # -----------------------------------------------------------------
    # encoder = model_dict[args.backbone]()
    # model = UniSiam(encoder=encoder, lamb=args.lamb, temp=args.temp,
    #                 dim_hidden=args.out_dim, use_transformers=use_transformers)
    # model.encoder = torch.nn.DataParallel(model.encoder)
    # model = model.cuda()

    # -----------------------------------------------------------------

    # if use_transformers:
    #     student = model_dict[args.backbone](
    #         patch_size=args.patch_size,
    #         drop_path_rate=args.drop_path,
    #         return_all_tokens=True,
    #         masked_im_modeling=args.use_masked_im_modeling,
    #     )
    #     teacher = model_dict[args.backbone](
    #         patch_size=args.patch_size,
    #         return_all_tokens=True,
    #     )
    #     embed_dim = student.embed_dim
    # else:
    #     student = model_dict[args.backbone]()
    #     teacher = copy.deepcopy(student)
    #     embed_dim = student.out_dim

    # # synchronize batch norms (if any)
    # if has_batchnorms(student):
    #     student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    #     teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    # # we need DDP wrapper to have synchro batch norms working...
    # teacher = nn.parallel.DistributedDataParallel(teacher)
    # student = nn.parallel.DistributedDataParallel(student)

    # # teacher and student start with the same weights
    # teacher.module.load_state_dict(
    #     student.module.state_dict(), strict=False)

    # print("AAA:{}".format(str(teacher.module.state_dict())
    #       == str(student.module.state_dict())))

    # # there is no backpropagation through the teacher, so no need for gradients
    # for p in teacher.parameters():
    #     p.requires_grad = False

    # model = MSiam(student=student, teacher=teacher, dim_in=embed_dim, args=args,
    #               lamb=args.lamb, temp=args.temp, use_transformers=use_transformers)

    # # model.student = torch.nn.DataParallel(model.student)
    # # model.teacher = torch.nn.DataParallel(model.teacher)

    model = model.cuda()
    print(model)

    return model


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
            data_path=args.data_path,
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


def adjust_learning_rate(args, optimizer, cur_iter, total_iter):
    lr = args.lr
    eta_min = lr * 1e-3
    lr = eta_min + (lr - eta_min) * \
        (1 + math.cos(math.pi * cur_iter / total_iter)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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


def save_model(model, epoch, loss, optimizer, batch_size, save_file, fp16_scaler=None):
    print('==> Saving... \n')
    save_state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'fp16_scaler': fp16_scaler
    }
    torch.save(save_state, save_file)
    del save_state


def save_student_teacher(student, teacher, epoch, loss, optimizer, batch_size,
                         save_file, teacher_memory, student_memory,
                         student_proj_memory, fp16_scaler=None):
    print('==> Saving... \n')
    save_state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'fp16_scaler': fp16_scaler,
        'teacher_memory': (teacher_memory.bank, teacher_memory.bank_ptr),
        'student_memory': (student_memory.bank, student_memory.bank_ptr),
        'student_proj_memory': (student_proj_memory.bank, student_proj_memory.bank_ptr)
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


def load_model(model, optimizer, fp16_scaler, ckpt_path):
    print('==> Loading... \n')
    # open checkpoint file
    checkpoint = torch.load(ckpt_path)

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

    del checkpoint
    return model, optimizer, fp16_scaler, epoch, loss, batch_size


def load_student_teacher(student, teacher, ckpt_path, teacher_memory=None,
                         student_memory=None, student_proj_memory=None,
                         optimizer=None, fp16_scaler=None):
    print('==> Loading... \n')
    # open checkpoint file

    checkpoint = torch.load(ckpt_path)

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    batch_size = checkpoint['batch_size']

    student.load_state_dict(checkpoint['student'])
    teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

    teacher_memory.load_memory_bank(
        checkpoint['teacher_memory'][0], checkpoint['teacher_memory'][1])
    student_memory.load_memory_bank(
        checkpoint['student_memory'][0], checkpoint['student_memory'][1])
    student_proj_memory.load_memory_bank(
        checkpoint['student_proj_memory'][0], checkpoint['student_proj_memory'][1])

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


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


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
