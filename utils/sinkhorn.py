# This code is adopted and modified from https://github.com/ojss/SAMPTransfer

import torch
from torch import distributed as dist
from torch import nn
import numpy as np


class Sinkhorn(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Arguments:
        - eps (float): regularization coefficient
        - max_iter (int): maximum number of Sinkhorn iterations
        - reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, eps_parameter, max_iter, thresh, reduction="none", device="cpu"):
        super(Sinkhorn, self).__init__()
        self.device = device
        self.eps_parameter = eps_parameter

        self.eps = eps
        if self.eps_parameter:
            self.eps = nn.Parameter(torch.tensor(self.eps))

        self.max_iter = max_iter
        self.thresh = thresh
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        cost_normalization = C.max()
        C = (
            C / cost_normalization
        )  # Needs to normalize the matrix to be consistent with reg

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / x_points).squeeze().to(self.device)

        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(
            1.0 / y_points).squeeze().to(self.device)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u,
                                      v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < self.thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


@torch.no_grad()
def distributed_sinkhorn(input, epsilon, iterations):
    # ensure that input has not infinite values
    np_input = input.cpu().numpy()
    input[input == float("Inf")] = np.nanmax(np_input[np_input != np.inf]) + 10

    # in case really high values are present in the input, gradually relax the
    # OT problem, by increasing epsilon (In practice: a value of 0.05 should be
    # enough to exit this loop)
    while True:
        # Q is K-by-B
        Q = torch.exp(input / epsilon).t()
        B = Q.shape[1]   # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        if not torch.isnan(Q).any():
            break
        epsilon += 0.1

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
