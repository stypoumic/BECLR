
import torch
import torch.nn as nn
import torch.nn.functional as F


class BECLR(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 dim_in: int,
                 args: dict,
                 is_teacher: bool = False):
        """
        Initializes a BECLR network for either student or teacher.

        Arguments:
            - encoder (nn.Module): the backbone encoder module
            - dim_in (int): input dimension of projection heads
            - args (dict): parsed keyword arguments for training
            - is_teacher (bool): specifies if it is the teacher network (optional)
        """
        super(BECLR, self).__init__()

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

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass of a given training batch through the 
        BECLR network.

        Arguments:
            - x (torch.Tensor): input batch of training images

        Returns:
            - student/teacher embeddings after the projection/ prediction head.
        """
        f = self.encoder(x)
        if self.is_teacher:
            z = self.proj(f)
            return z
        else:
            z = self.proj(f)
            p = self.pred(z)
            return p, z


class BECLRLoss(nn.Module):
    def __init__(self, args: dict, lamb_neg: float = 0.1, temp: float = 2.0):
        """
        Initializes the contrastive loss of BECLR.

        Arguments:
            - args (dict): parsed keyword arguments for training
            - lamb_neg (float): weight of negative loss term (optional)
            - temp (float): temperature parameter (optional)
        """
        super(BECLRLoss, self).__init__()
        self.args = args
        self.lamb_neg = lamb_neg
        self.temp = temp

    def forward(self,
                z_teacher: torch.Tensor,
                p_student: torch.Tensor,
                z_student: torch.Tensor,
                args: dict,
                epoch: int = None,
                memory: nn.Module = None):
        """
        Performs a forward pass of the current training batch through 
        the BECLR contrastive loss.

        Arguments:
            - x (torch.Tensor): Input batch of training images
            - z_teacher (torch.Tensor): teacher projection embeddings
            - p_student (torch.Tensor): student prediction embeddings
            - z_student (torch.Tensor): student projection embeddings
            - args (dict): parsed keyword arguments for training
            - epoch (int): current training epoch (optional)
            - memory (nn.Module): DyCE memory module (optional)

        Returns:
            - the contrastive loss value for the current batch.
        """
        bsz = args.batch_size
        # the enhanced batch is only used after args.memory_start_epoch
        if epoch >= args.memory_start_epoch and args.enhance_batch and not args.use_nnclr:
            bsz = args.batch_size * (1 + args.topk)

        # split into the two augmented views
        z1, z2 = torch.split(z_teacher, [bsz, bsz], dim=0)
        z1_s, z2_s = torch.split(z_student, [bsz, bsz], dim=0)
        p1, p2 = torch.split(p_student, [bsz, bsz], dim=0)

        # apply positive los term
        loss_pos = (self.pos(p1, z2)+self.pos(p2, z1))/2

        if args.uniformity_config != "TT":
            z1 = z1_s
            if args.uniformity_config == "SS":
                z2 = z2_s

        # apply negative loss term
        # (optionally): enhance number of negatives from DyCE memory
        if self.args.use_memory_in_loss:
            loss_neg = self.neg(z1, z2, epoch, memory, args.pos_threshold)
        else:
            loss_neg = self.neg(z1, z2)

        loss = loss_pos + self.lamb_neg * loss_neg
        std = self.std(z_teacher)
        loss_state = {
            'loss': loss,
            'loss_pos': loss_pos,
            'loss_neg': loss_neg,
            'std': std
        }
        return loss_state

    @torch.no_grad()
    def std(self, z: torch.Tensor):
        return torch.std(F.normalize(z, dim=1), dim=0).mean()

    def pos(self, p: torch.Tensor, z: torch.Tensor):
        """
        Calculates positive loss term.

        Arguments:
            - z (torch.Tensor): teacher projection embeddings for first view
            - p (torch.Tensor): student prediction embeddings for second view

        Returns:
            - the positive loss value.
        """
        z = z.detach()
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return -(p*z).sum(dim=1).mean()

    def neg(self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            epoch: int = None,
            memory: nn.Module = None,
            pos_threshold: int = 0.8):
        """
        Calculates negative loss term.

        Arguments:
            - z1 (torch.Tensor): projection embeddings for first view
            - z2 (torch.Tensor): prediction embeddings for second view
            - epoch (int): current training epoch (optional)
            - memory (nn.Module, optional): DyCE memory module (optional)
            - pos_threshold (float): threshold for choosing negatives, when 
                using the DyCe memory (optional)

        Returns:
            - the negative loss value.
        """
        z = torch.cat((z1, z2), 0)

        batch_size = z.shape[0] // 2
        n_neg = z.shape[0] - 2
        z = F.normalize(z, dim=-1)
        # mask out positives
        mask = 1-torch.eye(batch_size, dtype=z.dtype,
                           device=z.device).repeat(2, 2)
        if memory == None or epoch < self.args.memory_start_epoch:
            out = torch.matmul(z, z.T) * mask
            return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()
        else:
            out = torch.matmul(z, memory)
            # calculate number of negatives
            n_neg = torch.sum(out <= pos_threshold)
            # only keep memory embeddings above the positive similarity threshold
            out[out > pos_threshold] = 0.0
            return (out.div(self.temp).exp().sum(1)).div(n_neg).mean().log()
