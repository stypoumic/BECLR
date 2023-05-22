import torch


def log_sum_exp(u: torch.Tensor, dim: int):
    # Reduce log sum exp along axis
    u_max, __ = u.max(dim=dim, keepdim=True)
    log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
    return log_sum_exp_u


def log_sinkhorn(M: torch.Tensor, reg: float, num_iters: int):
    """
    Log-space-sinkhorn algorithm for better stability.
    """
    if M.dim() > 2:
        return batched_log_sinkhorn(M=M, reg=reg, num_iters=num_iters)

    # Initialize dual variable v (u is implicitly defined in the loop)
    # ==torch.log(torch.ones(m.size()[1]))
    log_v = torch.zeros(M.size()[1]).to(M.device)

    # Exponentiate the pairwise distance matrix
    log_K = -M / reg

    # Main loop
    for i in range(num_iters):
        # Match r marginals
        log_u = - log_sum_exp(log_K + log_v[None, :], dim=1)

        # Match c marginals
        log_v = - log_sum_exp(log_u[:, None] + log_K, dim=0)

    # Compute optimal plan, cost, return everything
    log_P = log_u[:, None] + log_K + log_v[None, :]
    return log_P


def batched_log_sinkhorn(M, reg: float, num_iters: int):
    """
    Batched version of log-space-sinkhorn.
    """
    batch_size, x_points, _ = M.shape
    # both marginals are fixed with equal weights
    mu = torch.empty(batch_size, x_points, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / x_points).squeeze().to(M.device)
    nu = torch.empty(batch_size, x_points, dtype=torch.float,
                     requires_grad=False).fill_(1.0 / x_points).squeeze().to(M.device)

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)
    # To check if algorithm terminates because of threshold
    # or max iterations reached
    actual_nits = 0
    # Stopping criterion
    thresh = 1e-1

    def C(M, u, v, reg):
        """Modified cost for logarithmic updates"""
        return (-M + u.unsqueeze(-1) + v.unsqueeze(-2)) / reg

    # Sinkhorn iterations
    for i in range(num_iters):
        u1 = u  # useful to check the update
        u = reg * (torch.log(mu + 1e-8) -
                   torch.logsumexp(C(M, u, v, reg), dim=-1)) + u
        v = reg * (torch.log(nu + 1e-8) - torch.logsumexp(C(M,
                   u, v, reg).transpose(-2, -1), dim=-1)) + v
        err = (u - u1).abs().sum(-1).mean()

        actual_nits += 1
        if err.item() < thresh:
            break

    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    log_p = C(M, U, V, reg)
    return log_p


class SOT(object):
    supported_distances = ['cosine', 'euclidean']

    def __init__(self, distance_metric: str = 'cosine', ot_reg: float = 0.1, sinkhorn_iterations: int = 20,
                 sigmoid: bool = False, mask_diag: bool = True, max_scale: bool = True):
        """
        :param distance_metric - Compute the cost matrix.
        :param ot_reg - Sinkhorn entropy regularization (lambda). For few-shot classification, 0.1-0.2 works best.
        :param sinkhorn_iterations - Maximum number of sinkhorn iterations.
        :param sigmoid - If to apply sigmoid(log_p) instead of the usual exp(log_p).
        :param mask_diag - Set to true to apply diagonal masking before and after the OT.
        :param max_scale - Re-scale the SOT values to range [0,1].
        """
        super().__init__()

        assert distance_metric.lower() in SOT.supported_distances and sinkhorn_iterations > 0

        self.sinkhorn_iterations = sinkhorn_iterations
        self.distance_metric = distance_metric.lower()
        self.mask_diag = mask_diag
        self.sigmoid = sigmoid
        self.ot_reg = ot_reg
        self.max_scale = max_scale
        self.diagonal_val = 1e3                         # value to mask self-values with

    def compute_cost(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute cost matrix.
        """
        if self.distance_metric == 'euclidean':
            M = torch.cdist(X, X, p=2)
            # scale euclidean distances to [0, 1]
            return M / M.max()

        elif self.distance_metric == 'cosine':
            # cosine distance
            return 1 - SOT.cosine_similarity(X)

    def mask_diagonal(self, M: torch.Tensor, value: float):
        """
        Set new value at a diagonal matrix.
        """
        if self.mask_diag:
            if M.dim() > 2:
                M[torch.eye(M.shape[1]).repeat(
                    M.shape[0], 1, 1).bool()] = value
            else:
                M.fill_diagonal_(value)
        return M

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the SOT features for X
        """
        # get masked cost matrix
        C = self.compute_cost(X=X)
        M = self.mask_diagonal(C, value=self.diagonal_val)

        # compute self-OT
        z_log = log_sinkhorn(M=M, reg=self.ot_reg,
                             num_iters=self.sinkhorn_iterations)

        if self.sigmoid:
            z = torch.sigmoid(z_log)
        else:
            z = torch.exp(z_log)

        # divide the SOT matrix by its max to scale it up
        if self.max_scale:
            z_max = z.max().item() if z.dim() <= 2 else z.amax(dim=(1, 2), keepdim=True)
            z = z / z_max

        # set self-values to 1
        return self.mask_diagonal(z, value=1)

    @staticmethod
    def cosine_similarity(a: torch.Tensor, eps: float = 1e-8):
        """
        Compute the pairwise cosine similarity between a matrix to itself.
        """
        d_n = a / a.norm(dim=-1, keepdim=True)
        if len(a.shape) > 2:
            C = torch.bmm(d_n, d_n.transpose(1, 2))
        else:
            C = torch.mm(d_n, d_n.transpose(0, 1))
        return C
