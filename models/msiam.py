# Implementation adapted from:
#     https://github.com/bbbdylan/unisiam
# under the MIT license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .head import iBOTHead
import numpy as np
from torch.cuda import comm
from torchmetrics.functional import pairwise_euclidean_distance


class CustomSequential(nn.Sequential):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d,
                nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False,
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


class AttFex(nn.Module):
    def __init__(self, args, wm_channels=64, wn_channels=32):
        super(AttFex, self).__init__()

        act_fn = nn.LeakyReLU(0.2)

        ## AttFEX Module ##
        # 1x1 Convs representing M(.), N(.)
        # self.n = args.batch_size * 2
        self.n = int((args.batch_size * (1 + args.topk) * 2))
        #
        #  torch.cuda.device_count())
        # self.n = int(args.batch_size * 2 / torch.cuda.device_count())

        self.fe = nn.Sequential(
            nn.Conv2d(in_channels=self.n, out_channels=wm_channels, kernel_size=(  # 64 --> args.wm
                1, 1), stride=(1, 1), padding='valid', bias=False),
            act_fn,

            nn.Conv2d(in_channels=wm_channels, out_channels=wn_channels, kernel_size=(  # 64 --> args.wm, 32 --> args.wn
                1, 1), stride=(1, 1), padding='valid', bias=False),
            act_fn)

        # Query, Key and Value extractors as 1x1 Convs
        self.f_q = nn.Conv2d(in_channels=wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_k = nn.Conv2d(in_channels=wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)
        self.f_v = nn.Conv2d(in_channels=wn_channels, out_channels=1, kernel_size=(  # 32 --> args.wn
            1, 1), stride=(1, 1), padding='valid', bias=False)

    def forward(self, x):
        # print(x.size())
        # bsz = x.shape[0]
        # x_gathered = nn.parallel.gather(x, torch.cuda.device(0))
        # x_gathered = torch.reshape(x_gathered, (bsz, -1))
        # print(x.size())
        # if torch.any(x_gathered != x):
        #     print("AAA")
        # x = x_gathered

        # test2 = nn.parallel.scatter(test, [torch.cuda.device(i)
        #                                    for i in range(torch.cuda.device_count())])

        x = x[:, :, None, None]
        G = x.permute(1, 0, 2, 3)

        G = self.fe(G)

        xq = self.f_q(G)
        xk = self.f_k(G)
        xv = self.f_v(G)

        xq = xq.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])
        xk = xk.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])
        xv = xv.squeeze(dim=1).squeeze(dim=1).transpose(
            0, 1).reshape(-1, x.shape[2], x.shape[3])

        # Attention Block
        xq = xq.reshape(xq.shape[0], xq.shape[1]*xq.shape[2])
        xk = xk.reshape(xk.shape[0], xk.shape[1]*xk.shape[2])
        xv = xv.reshape(xv.shape[0], xv.shape[1]*xv.shape[2])

        G = torch.mm(xq, xk.transpose(0, 1)/xk.shape[1]**0.5)
        softmax = nn.Softmax(dim=-1)
        G = softmax(G)
        G = torch.mm(G, xv)

        # Transductive Mask transformed input
        G = G.reshape(-1, x.shape[2], x.shape[3])
        x = x * G
        x = nn.Flatten()(x)
        return x


class MSiam(nn.Module):
    def __init__(self, encoder, dim_in, args, use_transformers=False, is_teacher=False):
        super(MSiam, self).__init__()

        self.encoder = encoder

        dim_out = dim_in if args.out_dim is None else args.out_dim
        self.use_features_before_proj = False
        self.use_transformers = use_transformers
        self.use_feature_align = args.use_feature_align
        self.use_feature_align_teacher = args.use_feature_align_teacher
        self.is_teacher = is_teacher
        self.dist = args.dist

        if self.use_feature_align:
            self.feature_extractor = AttFex(args)
            self.proj_align = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
            )
            self.pred_align = nn.Sequential(
                nn.Linear(dim_out, dim_out//4),
                nn.BatchNorm1d(dim_out//4),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out//4, dim_out)
            )

        if self.use_transformers:
            # CustomSequential
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
            # # # multi-crop wrapper handles forward with inputs of different resolutions
            # if self.is_teacher:
            #     self.proj = iBOTHead(
            #         dim_in,
            #         args.out_dim,
            #         patch_out_dim=args.patch_out_dim,
            #         norm=args.norm_in_head,
            #         act=args.act_in_head,
            #         shared_head=args.shared_head_teacher,
            #     )
            # else:
            #     self.proj = iBOTHead(
            #         dim_in,
            #         args.out_dim,
            #         patch_out_dim=args.patch_out_dim,
            #         norm=args.norm_in_head,
            #         act=args.act_in_head,
            #         norm_last_layer=args.norm_last_layer,
            #         shared_head=args.shared_head,
            #     )
        else:
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
            if self.dist:
                self.pred_dist = nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.BatchNorm1d(dim_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_out, dim_out),
                    nn.BatchNorm1d(dim_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_out, dim_out),
                    nn.BatchNorm1d(dim_out),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_out, dim_out//4),
                    nn.BatchNorm1d(dim_out//4),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_out//4, dim_out)
                )
            # prototype layer
            # self.prototypes = None
            # if args.use_clustering:
            #     self.prototypes = nn.Linear(
            #         dim_out, args.nmb_prototypes, bias=False)

    def forward(self, x, masks=None):

        if self.use_transformers:
            if self.is_teacher:
                teacher_out = self.encoder(x)
                f_cls = teacher_out[:, 0]
                f_patch = teacher_out[:, 1:]
                z = self.proj(f_cls)
                #############################
                # teacher_out = self.proj(teacher_out)
                # z = teacher_out[:, 0]
                # f_patch = teacher_out[:, 1:]
                #############################
                # z, f_patch = self.proj(teacher_out)
                return z, f_patch
            else:
                student_out = self.encoder(x, mask=masks)
                # ---------UniHead no patches
                f_cls = student_out[:, 0]
                f_patch = student_out[:, 1:]
                z = self.proj(f_cls)
                p = self.pred(z)
                #############################
                # ---------UniHead patches
                # student_out = self.pred(self.proj(student_out))
                # p = student_out[:, 0]
                # f_patch = student_out[:, 1:]
                ############################
                # ---------iBoT head
                # p, f_patch = self.proj(student_out)
                return p, f_patch, z
        else:
            f = self.encoder(x)
            if not self.is_teacher:
                # f = torch.flatten(f, 1)
                z = self.proj(f)
                p = self.pred(z)

                # apply self-distilliation
                if self.dist:
                    p_dist = self.pred_dist(f)
                else:
                    p_dist = None

                # # get prototypes
                # if self.prototypes != None:
                #     prototypes = self.prototypes(p)
                # else:
                #     prototypes = None

                return p, z, p_dist
            else:
                z = self.proj(f)

                # # get prototypes
                # if self.prototypes != None:
                #     prototypes = self.prototypes(z)
                # else:
                #     prototypes = None

                return z


class MSiamLoss(nn.Module):

    def __init__(self, args, lamb_neg=0.1, lamb_patch=0.1, temp=2.0, ngcrops=2, student_temp=0.1,
                 center_momentum=0.9, center_momentum2=0.9, lambda1=1):
        super(MSiamLoss, self).__init__()
        self.args = args
        self.use_patches = args.use_patches
        self.use_transformers = 'deit' in args.backbone
        self.lamb_neg = args.lamb_neg
        self.lamb_patch = lamb_patch
        self.temp = temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.lambda1 = lambda1

    def forward(self, z_teacher, p_student, z_student, args, epoch=None,
                p_refined=None, z_refined=None, p_dist=None, z_dist=None, w_ori=1.0,
                w_dist=0.5, memory=None):

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

        # if self.use_transformers:
        #     loss_neg = self.neg(
        #         torch.cat((torch.unsqueeze(teacher_cls, 1), teacher_patch), dim=1))
        # else:
        # changed to student (student_z)``
        if args.uniformity_config != "TT":
            z1 = z1_s
            if args.uniformity_config == "SS":
                z2 = z2_s

        if self.args.use_memory_in_loss:
            loss_neg = self.neg(z1, z2, epoch, memory, args.pos_threshold)
        else:
            loss_neg = self.neg(z1, z2)

        if p_refined is not None:
            p1_refined, p2_refined = torch.split(
                p_refined, [p_bsz, p_bsz], dim=0)
            if z_refined is not None:
                z1_refined, z2_refined = torch.split(
                    z_refined, [z_bsz, z_bsz], dim=0)
            else:
                z1_refined, z2_refined = z1, z2

            loss_pos_ref = (self.pos(p1_refined, z2_refined) +
                            self.pos(p2_refined, z1_refined))/2

            loss = (loss_pos * w_ori + loss_pos_ref) / (1 + w_ori)
        else:
            loss_pos_ref = 0.0
            loss = loss_pos

        loss = loss + self.lamb_neg * loss_neg

        if z_dist is not None:
            loss_dist = self.pos(p_dist, z_dist)
            loss = w_dist * loss + w_dist * loss_dist
        else:
            loss_dist = 0.0
        loss_patch = 0.0

        std = self.std(z_teacher)

        loss_state = {
            'loss': loss,
            'loss_pos': loss_pos,
            'loss_neg': loss_neg,
            'loss_patch': loss_patch,
            'loss_dist': loss_dist,
            'loss_pos_ref': loss_pos_ref,
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
        # if self.use_transformers:
        #     z1 = z.permute(1, 0, 2)
        #     z2 = z.permute(1, 2, 0)
        #     out = torch.matmul(z1, z2) * mask
        # else:
        if memory == None or epoch < self.args.memory_start_epoch:
            out = torch.matmul(z, z.T) * mask
            return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()
        else:
            out = torch.matmul(z, memory)
            n_neg = torch.sum(out <= pos_threshold)
            out[out > pos_threshold] = 0.0
            return (out.div(self.temp).exp().sum(1)).div(n_neg).mean().log()

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + \
            cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / \
            (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + \
            patch_center * (1 - self.center_momentum2)


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


def swav_loss(args, student_out, teacher_out, student, teacher, s_memory, t_memory):
    # ============ swav loss ... ============
    loss = 0
    bsz = teacher_out.shape[0] // 2
    z1, z2 = torch.split(teacher_out, [bsz, bsz], dim=0)
    p1, p2 = torch.split(student_out, [bsz, bsz], dim=0)

    # p2 --> z1
    # p1 --> z2
    # z2 --> p1
    # z1 --> p2
    loss = code_predict(args, z1, t_memory, teacher, p2) + code_predict(args, z2, t_memory, teacher, p1) + \
        code_predict(args, p1, s_memory, student, z2) + \
        code_predict(args, p2, s_memory, student, z1)
    loss /= 4

    return loss


def code_predict(args, output, memory, model, x):
    # concat to output X embeddings from memory
    N = 4 * args.batch_size
    bs = output.shape[0]

    # load memory
    bank = memory.bank.cuda()

    # get N random embeddings from memory
    indices = torch.randperm(bank.shape[1])[:N]
    # multiply with prototypes weights to get memory prototypes
    memory_prototypes = torch.mm(
        bank[:, indices].T, model.module.prototypes.weight.t())
    # concat with orignal prototypes from original features
    output = torch.cat((memory_prototypes, output), 0)

    # get assignments (only for bs)
    q = distributed_sinkhorn(output, args)[-bs:]

    # cluster assignment prediction
    return -torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))


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
