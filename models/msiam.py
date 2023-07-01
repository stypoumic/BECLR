# Implementation adapted from:
#     https://github.com/bbbdylan/unisiam
# under the MIT license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.cuda import comm
from torchmetrics.functional import pairwise_euclidean_distance


class MSiam(nn.Module):
    def __init__(self, encoder, dim_in, args, is_teacher=False):
        super(MSiam, self).__init__()

        self.encoder = encoder

        dim_out = dim_in if args.out_dim is None else args.out_dim
        self.is_teacher = is_teacher

        self.encoder.fc = None
        self.proj = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out),
        )
        self.pred = nn.Sequential(
            nn.Linear(dim_out, dim_out//4),
            nn.BatchNorm1d(dim_out//4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out//4, dim_out)
        )

    def forward(self, x, masks=None):
        f = self.encoder(x)
        if not self.is_teacher:
            z = self.proj(f)
            p = self.pred(z)

            return p, z
        else:
            z = self.proj(f)

            return z


class MSiamLoss(nn.Module):

    def __init__(self, args, lamb_neg=0.1, temp=2.0):
        super(MSiamLoss, self).__init__()
        self.args = args
        self.lamb_neg = args.lamb_neg
        self.temp = temp

    def forward(self, z_teacher, p_student, z_student, args, epoch=None, memory=None):

        z_bsz = p_bsz = args.batch_size
        if epoch >= args.memory_start_epoch:
            if args.enhance_batch:
                z_bsz = p_bsz = args.batch_size * (1 + args.topk)
            elif args.use_nnclr:
                z_bsz = args.batch_size * (args.topk)

        z1, z2 = torch.split(z_teacher, [z_bsz, z_bsz], dim=0)
        z1_s, z2_s = torch.split(z_student, [p_bsz, p_bsz], dim=0)
        p1, p2 = torch.split(p_student, [p_bsz, p_bsz], dim=0)
        if z_bsz != p_bsz:
            p1 = p1.repeat(args.topk, 1)
            p2 = p2.repeat(args.topk, 1)

        loss_pos = (self.pos(p1, z2)+self.pos(p2, z1))/2

        if args.uniformity_config != "TT":
            z1 = z1_s
            if args.uniformity_config == "SS":
                z2 = z2_s

        if self.args.use_memory_in_loss:
            loss_neg = self.neg(z1, z2, epoch, memory, args.pos_threshold)
        else:
            loss_neg = self.neg(z1, z2)

        loss = loss_pos

        loss = loss + self.lamb_neg * loss_neg

        loss_patch = 0.0

        std = self.std(z_teacher)

        loss_state = {
            'loss': loss,
            'loss_pos': loss_pos,
            'loss_neg': loss_neg,
            'loss_patch': loss_patch,
            'std': std
        }
        return loss_state

    @torch.no_grad()
    def std(self, z):
        return torch.std(F.normalize(z, dim=1), dim=0).mean()

    def pos(self, p, z):
        z = z.detach()
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return -(p*z).sum(dim=1).mean()

    def neg(self, z1, z2, epoch=None, memory=None, pos_threshold=0.8):
        z = torch.cat((z1, z2), 0)

        batch_size = z.shape[0] // 2
        n_neg = z.shape[0] - 2
        z = F.normalize(z, dim=-1)
        mask = 1-torch.eye(batch_size, dtype=z.dtype,
                           device=z.device).repeat(2, 2)
        if memory == None or epoch < self.args.memory_start_epoch:
            out = torch.matmul(z, z.T) * mask
            return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()
        else:
            out = torch.matmul(z, memory)
            n_neg = torch.sum(out <= pos_threshold)
            out[out > pos_threshold] = 0.0
            return (out.div(self.temp).exp().sum(1)).div(n_neg).mean().log()


class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def cross_entropy(self, x1, x2):
        # return -torch.mean(torch.sum(x1 * F.log_softmax(x2, dim=1), dim=1))
        return -torch.mean(torch.sum(x1 * torch.log(x2), dim=1))
        # return -torch.mean(torch.sum(x1 * x2, dim=1))

    def forward(self, args, p, z, memory):
        # get cluster prototypes
        centers = memory.centers.clone().cuda()

        # Normalize
        z = torch.nn.functional.normalize(z, dim=1)
        p = torch.nn.functional.normalize(p, dim=1)
        centers = torch.nn.functional.normalize(centers, dim=1)

        # split embeddings into 2 views
        bsz = z.shape[0] // 2
        z1, z2 = torch.split(z, [bsz, bsz], dim=0)
        p1, p2 = torch.split(p, [bsz, bsz], dim=0)

        # create cost matrix between student/teacher embeddings & cluster centers
        za_1 = pairwise_euclidean_distance(z1, centers)
        za_2 = pairwise_euclidean_distance(z2, centers)
        pa_1 = pairwise_euclidean_distance(p1, centers)
        pa_2 = pairwise_euclidean_distance(p2, centers)

        # apply optimal transport to get assignments (# BS x K)
        if args.cls_use_enhanced_batch:
            za_1 = distributed_sinkhorn(
                za_1, args.epsilon, args.sinkhorn_iterations)
            za_2 = distributed_sinkhorn(
                za_2, args.epsilon, args.sinkhorn_iterations)
            pa_1 = distributed_sinkhorn(
                pa_1, args.epsilon, args.sinkhorn_iterations)
            pa_2 = distributed_sinkhorn(
                pa_2, args.epsilon, args.sinkhorn_iterations)
        else:
            za_1 = distributed_sinkhorn(za_1, args.epsilon, args.sinkhorn_iterations)[
                :args.batch_size]
            za_2 = distributed_sinkhorn(za_2, args.epsilon, args.sinkhorn_iterations)[
                :args.batch_size]
            pa_1 = distributed_sinkhorn(pa_1, args.epsilon, args.sinkhorn_iterations)[
                :args.batch_size]
            pa_2 = distributed_sinkhorn(pa_2, args.epsilon, args.sinkhorn_iterations)[
                :args.batch_size]

        # print(pa_1.requires_grad)
        # print(self.cross_entropy(za_1, pa_1).requires_grad)

        if args.cls_use_both_views:
            return (self.cross_entropy(za_1, pa_1) + self.cross_entropy(za_1, pa_2) +
                    self.cross_entropy(za_2, pa_1) + self.cross_entropy(za_2, pa_2)) / 4
        else:
            return (self.cross_entropy(za_1, pa_2) + self.cross_entropy(za_2, pa_1)) / 2


# @torch.no_grad()
def distributed_sinkhorn(out, epsilon, iterations):
    # Q is K-by-B for consistency with notations from our paper
    Q0 = torch.exp(out / epsilon).t()
    B = Q0.shape[1]   # number of samples to assign
    K = Q0.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q0)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q = Q0 / sum_Q

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q1 = (Q / sum_of_rows) / K

        # normalize each column: total weight per sample must be 1/B
        Q2 = (Q1 / torch.sum(Q1, dim=0, keepdim=True)) / B

    # Q *= B  # the colomns must sum to 1 so that Q is an assignment
    # print(torch.sum(Q, dim=1, keepdim=True))
    return (Q2 * B).t()
