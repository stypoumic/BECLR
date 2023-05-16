import torch
from torch import nn
from torch.nn import functional as F

from sinkhorn import Sinkhorn


class OptimalTransport(nn.Module):
    def __init__(
            self,
            regularization,
            max_iter,
            stopping_criterion,
            learn_regularization=False,
            power_transform=None,
            device="cpu"  # pass in the validation phase
    ):
        super(OptimalTransport, self).__init__()
        self.sinkhorn = Sinkhorn(
            eps=regularization,
            max_iter=max_iter,
            thresh=stopping_criterion,
            eps_parameter=learn_regularization,
            device=device
        )
        self.beta = power_transform

    def forward(self, z_support, z_query):
        """
        Applies Optimal Transport from query images to support images,
            and uses the outcome to predict query classification scores.
        Args:
            z_support (torch.Tensor): shape (number_of_support_set_images, feature_dim)
            z_query (torch.Tensor): shape (number_of_query_set_images, feature_dim)
        Returns:
            tuple(torch.Tensor, torch.Tensor) : resp. transported support set features,
                and unmodified query set features
        """
        if self.beta:
            z_support = F.normalize(torch.pow(z_support + 1e-6, self.beta))
            z_query = F.normalize(torch.pow(z_query + 1e-6, self.beta))

        _, transport_plan, _ = self.sinkhorn(z_support, z_query)

        z_support_transported = torch.matmul(
            transport_plan / transport_plan.sum(axis=1, keepdims=True), z_query
        )

        return z_support_transported, z_query
