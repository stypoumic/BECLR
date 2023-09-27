import torch
from torch import nn
from torch.nn import functional as F

from utils.sinkhorn import Sinkhorn


class OpTA(nn.Module):
    def __init__(
            self,
            regularization: float,
            max_iter: int,
            stopping_criterion: float,
            device: str = "cpu"):
        """
        Initializes an OpTA inference module.

        Arguments:
            - regularization (float): regularization coefficient
            - max_iter (int): maximum number of Sinkhorn iterations
            - stopping_criterion (float): threshold for Sinkhorn algorithm
            - device (str): device used for optimal transport calculations

        Returns:
            - tuple(transported support prototypes, unchanged query features)
        """
        super(OpTA, self).__init__()
        self.sinkhorn = Sinkhorn(
            eps=regularization,
            max_iter=max_iter,
            thresh=stopping_criterion,
            eps_parameter=False,
            device=device)

    def forward(self, z_support: torch.Tensor, z_query: torch.Tensor):
        """
        Applies Optimal Transport between support and query features.

        Arguments:
            - z_support (torch.Tensor): support prototypes (or features)
            - z_query (torch.Tensor): query features

        Returns:
            - tuple(transported support prototypes, unchanged query features)
        """
        cost, transport_plan, _ = self.sinkhorn(z_support, z_query)

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), z_query
        )

        return z_support_transported, z_query
