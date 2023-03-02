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


class MSiam(nn.Module):
    def __init__(self, encoder, dim_in, args, use_transformers=False, is_teacher=False):
        super(MSiam, self).__init__()

        self.encoder = encoder

        dim_out = dim_in if args.out_dim is None else args.out_dim
        self.use_transformers = use_transformers
        self.is_teacher = is_teacher

        if self.use_transformers:
            # multi-crop wrapper handles forward with inputs of different resolutions
            if self.is_teacher:
                self.proj = iBOTHead(
                    dim_in,
                    args.out_dim,
                    patch_out_dim=args.patch_out_dim,
                    norm=args.norm_in_head,
                    act=args.act_in_head,
                    shared_head=args.shared_head_teacher,
                ),
            else:
                self.proj = iBOTHead(
                    dim_in,
                    args.out_dim,
                    patch_out_dim=args.patch_out_dim,
                    norm=args.norm_in_head,
                    act=args.act_in_head,
                    norm_last_layer=args.norm_last_layer,
                    shared_head=args.shared_head,
                )
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

    def forward(self, x, masks=None):

        if self.use_transformers:
            # get cls and patch features
            # if self.is_teacher:
            #     f_cls, f_patch = self.proj(x)
            # else:
            #     f_cls, f_patch = self.proj(
            #         x, mask=masks)

            # return f_cls, f_patch
            # if self.is_teacher:
            #     f = self.encoder(x)
            #     z, t_patches = self.proj(f)
            #     return z, t_patches
            # else:
            #     f = self.encoder(x, mask=masks)
            #     z = self.proj(f)
            #     p, s_patches = self.pred(z)
            #     return p, s_patches

            if self.is_teacher:
                f = self.encoder(x)
                z = self.proj(f)
                return z, None
            else:
                f = self.encoder(x, mask=masks)
                p = self.proj(f)
                return p, None
        else:
            f = self.encoder(x)
            z = self.proj(f)
            if not self.is_teacher:
                p = self.pred(z)
                return p
            return z


class MSiamLoss(nn.Module):

    def __init__(self, args, lamb_neg=0.1, lamb_patch=0.1, temp=2.0, ngcrops=2, student_temp=0.1,
                 center_momentum=0.9, center_momentum2=0.9, lambda1=1.0):
        super(MSiamLoss, self).__init__()
        self.use_patches = args.use_patches
        self.use_transformers = 'deit' in args.backbone
        self.lamb_neg = lamb_neg
        self.lamb_patch = lamb_patch
        self.temp = temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops

        self.register_buffer("center", torch.zeros(1, args.out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, args.patch_out_dim))
        self.lambda1 = lambda1

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(args.warmup_teacher_temp,
                        args.teacher_temp, args.warmup_teacher_temp_epochs),
            np.ones(args.epochs - args.warmup_teacher_temp_epochs) *
            args.teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(args.warmup_teacher_patch_temp,
                        args.teacher_patch_temp, args.warmup_teacher_temp_epochs),
            np.ones(args.epochs - args.warmup_teacher_temp_epochs) *
            args.teacher_patch_temp
        ))

    def forward(self, teacher_cls, student_cls, bsz,
                teacher_patch=None, student_patch=None, student_mask=None,
                epoch=None):
        z1, z2 = torch.split(teacher_cls, [bsz, bsz], dim=0)
        p1, p2 = torch.split(student_cls, [bsz, bsz], dim=0)

        loss_pos = (self.pos(p1, z2)+self.pos(p2, z1))/2
        loss_neg = self.neg(teacher_cls)
        loss = loss_pos + self.lamb_neg * loss_neg

        if self.use_patches and self.use_transformers:
            # [CLS] and patch for global patches
            student_cls = student_cls / self.student_temp
            student_cls_c = student_cls.chunk(self.ngcrops)
            student_patch = student_patch / self.student_temp
            student_patch_c = student_patch.chunk(self.ngcrops)

            # teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch]
            temp2 = self.teacher_temp2_schedule[epoch]
            teacher_cls_c = F.softmax(
                (teacher_cls - self.center) / temp, dim=-1)
            teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
            teacher_patch_c = F.softmax(
                (teacher_patch - self.center2) / temp2, dim=-1)
            teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

            loss_patch, n_patch_loss_terms = 0.0, 0
            for q in range(len(teacher_cls_c)):
                for v in range(len(student_cls_c)):
                    loss_p = torch.sum(-teacher_patch_c[q] * F.log_softmax(
                        student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss_p = torch.sum(loss_p * mask.float(),
                                       dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    loss_patch += loss_p.mean()
                    n_patch_loss_terms += 1

            loss_patch = loss_patch / n_patch_loss_terms * self.lambda1
            self.update_center(teacher_cls, teacher_patch)
            loss += self.lamb_patch * loss_patch
        else:
            loss_patch = 0.0

        std = self.std(teacher_cls)

        return loss, loss_pos, loss_neg, loss_patch, std

    @torch.no_grad()
    def std(self, z):
        return torch.std(F.normalize(z, dim=1), dim=0).mean()

    def pos(self, p, z):
        z = z.detach()
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return -(p*z).sum(dim=1).mean()

    def neg(self, z):
        batch_size = z.shape[0] // 2
        n_neg = z.shape[0] - 2
        z = F.normalize(z, dim=-1)
        mask = 1-torch.eye(batch_size, dtype=z.dtype,
                           device=z.device).repeat(2, 2)
        out = torch.matmul(z, z.T) * mask
        return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()

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
