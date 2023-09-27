import hdbscan
import torch
import torch.nn.functional as F
from lightly.loss.memory_bank import MemoryBankModule
from sklearn import cluster
from torchmetrics.functional import pairwise_euclidean_distance

from utils.sinkhorn import distributed_sinkhorn
from utils.visualize import visualize_memory


class DyCE(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16, origin: str = None):
        """
        Initializes a DyCE memory module.

        Arguments:
            - size (int): the size of the memory space
            - origin (str): student or teacher network origin
        """
        super(DyCE, self).__init__(size)
        # register buffers for stored prototypes and labels in the memory
        self.register_buffer(
            "prototypes", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "labels", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )
        self.start_clustering = False
        self.last_cluster_epoch = 0
        self.last_vis_epoch = 0
        self.origin = origin

    def load_memory_bank(self, memory_bank: tuple):
        """
        Loads the state of a DyCE memory module.

        Arguments:
            - memory_bank (tuple): tuple containing the stored memory embeddings, 
                memory bank pointer, stored memoro prototypes, and stored 
                assignments for all embeddings
        """
        self.bank = memory_bank[0]
        self.bank_ptr = memory_bank[1]
        self.labels = memory_bank[2]
        self.prototypes = memory_bank[3]
        self.start_clustering = True

    @torch.no_grad()
    def cluster_memory_embeddings(self,
                                  cluster_algo: str = "kmeans",
                                  num_clusters: int = 300,
                                  min_cluster_size: int = 4):
        """
        Performs a clustering step either for initializing the memory protoypes 
        or periodically restarting their positions (if reclustering is enabled). 
        Stores the results internally in self.prototypes, self.lalbels.

        Arguments:
            - cluster_algo (str): choice of clustering algorithm (optional)
            - num_clusters (int): number of prototypes to create (optional)
            - min_cluster_size (int): minimum number of samples per cluster 
                in case of hdbscan clustering algorithm (optional)
        """
        bank = self.bank.T.detach()

        bank_np = F.normalize(bank.detach()).cpu().numpy()

        if cluster_algo == "kmeans":
            clf, labels, _ = clusterer(bank_np,
                                       n_clusters=num_clusters,
                                       algo="kmeans")
            # get cluster means & labels
            prototypes = clf.cluster_centers_
            prototypes = torch.from_numpy(prototypes).type_as(bank).cpu()
            labels = torch.from_numpy(labels).type_as(bank).long().cpu()
            # do not upddate the clusters in the memory in case of redundancy
            # from kmeans
            if self.start_clustering and len(labels.unique(return_counts=True)[-1]) < num_clusters:
                return
            self.prototypes = prototypes
            self.labels = labels
        else:
            # possibly reranking before clustering
            clf, labels, probs = clusterer(
                bank_np.astype(float),
                algo="hdbscan",
                metric="euclidean",
                hdb_min_cluster_size=min_cluster_size
            )
            labels = torch.from_numpy(labels).type_as(bank).long()

            if -1 in labels:
                # breakpoint()
                non_noise_indices = ~(labels == -1)
                labels_masked = labels.masked_select(
                    non_noise_indices
                )
                bank = bank.index_select(
                    0, non_noise_indices.nonzero().flatten()
                )

            # copies labels across feature dimension  (memory_size x feature_dim)
            tmp = labels_masked.view(
                labels_masked.size(0), 1).expand(-1, bank.size(1))

            unique_labels, labels_count = tmp.unique(dim=0, return_counts=True)
            # get cluster means
            prototypes = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
                0, tmp.type(torch.int64), bank)
            self.prototypes = prototypes / labels_count.float().unsqueeze(1).cpu()
            self.labels = labels.cpu()

    @torch.no_grad()
    def save_memory_embeddings(self,
                               args: dict,
                               z: torch.Tensor,
                               dist_metric: str = "euclidean",
                               momentum=0.9):
        """
        Finds the optimal assignments between current batch embeddings and stored
        memory prototypes and ensures equipartitioning of assignments. The optimal
        assignments are then used for updating the memory protoypes and  
        partition.

        Arguments:
            - args (dict): parsed keyword training arguments: 
            - z (torch.Tensor): the input batch embeddings
            - dist_metric (str): choice of distance metric for calculating 
                distance matrix for optimal transport (optional)
            - momentum (float): momentum parameter for updating the memory 
                prototypes (optional)
        """
        prototypes = self.prototypes.clone().cpu()

        if dist_metric == "cosine":
            # Normalize batch & memory embeddings
            z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
            prototypes_normed = torch.nn.functional.normalize(
                prototypes, dim=1).cuda()  # K x D

            # create cost matrix between batch embeddings & cluster prototypes
            Q = torch.einsum("nd,md->nm", z_normed,
                             prototypes_normed)  # BS x K
        else:
            if args.eucl_norm:
                # Normalize batch & memory embeddings
                z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
                prototypes_normed = torch.nn.functional.normalize(
                    prototypes, dim=1).cuda()  # K x D
                # create cost matrix between batch embeddings & cluster prototypes
                Q = pairwise_euclidean_distance(z_normed, prototypes_normed)
            else:
                # create cost matrix between batch embeddings & cluster prototypes
                Q = pairwise_euclidean_distance(z, prototypes.cuda())

        # apply optimal transport between batch embeddings and cluster prototypes
        Q = distributed_sinkhorn(
            Q, args.epsilon, args.sinkhorn_iterations)  # BS x K

        # get assignments (batch labels)
        batch_labels = torch.argmax(Q, dim=1)

        # add equipartitioned batch to memory and discard oldest embeddings
        batch_size = z.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = z[: self.size - ptr].T.detach()
            self.labels[ptr:] = batch_labels[: self.size - ptr].detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr: ptr + batch_size] = z.T.detach()
            self.labels[ptr: ptr + batch_size] = batch_labels.detach()
            self.bank_ptr[0] = ptr + batch_size

        # Update cluster prototypes
        labels = self.labels.clone().cpu()
        bank = self.bank.clone().cpu().T.detach()

        view = labels.view(labels.size(0), 1).expand(-1, bank.size(1))
        unique_labels, labels_count = view.unique(dim=0, return_counts=True)
        deleted_labels = []
        for i in range(0, prototypes.shape[0]):
            if i not in unique_labels[:, 0]:
                deleted_labels.append(i)
                label = torch.tensor([[i]]).expand(-1, bank.size(1))
                unique_labels = torch.cat((unique_labels, label), 0)
                labels_count = torch.cat(
                    (labels_count, torch.tensor([0.001])), 0)

        # get cluster means
        prototypes_next = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
            0, view.type(torch.int64), bank).cpu()  # UN x 512
        prototypes_next = prototypes_next / labels_count.float().unsqueeze(1)

        # in case a stored cluster ends up without any assignments, use the
        # previous value of its prototype as the new prototype
        for i in deleted_labels:
            prototypes_next[i, :] = prototypes[i, :]

        # EMA update of cluster prototypes
        self.prototypes = prototypes_next * \
            (1 - momentum) + momentum * prototypes

        return z, bank.T

    def get_NN(self,
               output: torch.Tensor,
               epoch: int,
               args):
        """
        Performs a forward pass when reproducing NNCLR.

        Arguments:
            - output (torch.Tensor): the input batch embeddings
            - epoch (int): the current training epoch
            - args (dict): parsed keyword training arguments: 

        Returns:
            - the NN memory embeddings for each input batch embedding
        """
        bsz = output.shape[0]

        # Add latest batch to the memory queue
        output, bank = super(DyCE, self).forward(
            output, None, update=True)
        bank = bank.to(output.device).t()

        # only return the nearest neighbor features instead of the originals,
        # in case the NNCLR start epoch has passed
        if epoch >= args.memory_start_epoch:
            # Normalize batch & memory embeddings
            output_normed = torch.nn.functional.normalize(output, dim=1)
            bank_normed = torch.nn.functional.normalize(bank, dim=1)

            # create similarity matrix between batch & memory embeddings
            similarity_matrix = torch.einsum(
                "nd,md->nm", output_normed, bank_normed)

            # find nearest-neighbor for each batch embedding
            index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
            output = torch.index_select(
                bank, dim=0, index=index_nearest_neighbours)

        return output

    def get_top_kNN(self,
                    output: torch.Tensor,
                    epoch: int,
                    args,
                    k: int = 5):
        """
        Performs a forward pass of DyCE.

        Arguments:
            - output (torch.Tensor): the input batch embeddings
            - epoch (int): the current training epoch
            - args (dict): parsed keyword training arguments: 
            - k (int): the number of top-k NNs to concat to the orginal batch 
                (optional)

        Returns:
            - the enhanced batch embeddings
        """
        # pointer of memory bank (to know when it fills for the first time)
        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0
        bsz = output.shape[0]

        if self.start_clustering == False:
            # if clusters not yet initialized
            if ptr + bsz >= self.size:
                # if memory is full for the first time
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)

                self.last_cluster_epoch = epoch
                self.start_clustering = True

                # Add latest batch to the memory queue using Optimal Transport
                output, bank = self.save_memory_embeddings(
                    args, output, dist_metric=args.memory_dist_metric,
                    momentum=args.memory_momentum)
                prototypes_initialized = True
            else:
                # Add latest batch to the memory (memory not yet full for first time)
                output, bank = super(DyCE, self).forward(
                    output, None, update=True)
                prototypes_initialized = False
        else:
            # cluster are now initialized
            if args.recluster and epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
                # periodically restart the memory clusters if reclustering is enabled
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo,
                    num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            if len(self.labels.unique(return_counts=True)[-1]) <= 1:
                # restart memory clusters in case the memory embeddings have
                # converged to a single cluster
                # (In practice: not used, but covers the case when not suitable
                # hyperparameters for the OT memory updating have been chosen)
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo,
                    num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            # visualize memory embeddings using UMAP with an args.visual_freq
            # epoch frequency
            if epoch % args.visual_freq == 0 and epoch != self.last_vis_epoch:
                self.last_vis_epoch = epoch
                if self.origin in ["teacher", "student"]:
                    visualize_memory(self, args.save_path,
                                     self.origin, epoch=epoch,
                                     n_samples=args.memory_scale)

            # Add latest batch to the memory queue using Optimal Transport
            output, bank = self.save_memory_embeddings(
                args, output, dist_metric=args.memory_dist_metric, momentum=args.memory_momentum)
            prototypes_initialized = True

        bank = bank.to(output.device).t()
        # only concat the nearest neighbor features in case the memory start
        # epoch has passed (i.e. the memory has converged to stable state)
        if epoch >= args.memory_start_epoch:
            # Normalize batch & memory embeddings
            output_normed = torch.nn.functional.normalize(output, dim=1)
            bank_normed = torch.nn.functional.normalize(bank, dim=1)

            # split embeddings of the 2 views
            z1, z2 = torch.split(
                output_normed, [args.batch_size, args.batch_size], dim=0)

            # create similarity matrix between batch & memory embeddings
            similarity_matrix1 = torch.einsum(
                "nd,md->nm", z1, bank_normed)
            similarity_matrix2 = torch.einsum(
                "nd,md->nm", z2, bank_normed)

            # if the stored prototypes are initialized
            if prototypes_initialized:
                prototypes = self.prototypes.clone().cuda()
                labels = self.labels.clone().cuda()

                # Normalize prototypes
                prototypes = torch.nn.functional.normalize(prototypes, dim=1)

                # create similarity matrix between batch embeddings & prototypes
                z_center_similarity_matrix_1 = torch.einsum(
                    "nd,md->nm", z1, prototypes)
                z_center_similarity_matrix_2 = torch.einsum(
                    "nd,md->nm", z2, prototypes)

                # find nearest prototypes for each batch embedding
                _, topk_clusters_1 = torch.topk(
                    z_center_similarity_matrix_1, 1, dim=1)
                _, topk_clusters_2 = torch.topk(
                    z_center_similarity_matrix_2, 1, dim=1)

                z1_final = z1.clone()
                z2_final = z2.clone()
                # for each batch embedding
                for i in range(topk_clusters_1.shape[0]):

                    clusters_1 = topk_clusters_1[i, :]
                    clusters_2 = topk_clusters_2[i, :]

                    # find memory embedding indices that belong to the selected
                    # nearest cluster/prototype (for each view)
                    indices_1 = (labels[..., None] ==
                                 clusters_1).any(-1).nonzero().squeeze()
                    indices_2 = (labels[..., None] ==
                                 clusters_2).any(-1).nonzero().squeeze()

                    if indices_1.nelement() == 0:
                        # sanity check that all of the selected clusters more
                        # than 0 assigned memory embeddings
                        _, topk_indices_1 = torch.topk(
                            similarity_matrix1[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding &
                        # selected partition embeddings
                        tmp = bank_normed[indices_1, :].unsqueeze(
                            0) if indices_1.nelement() == 1 else bank_normed[indices_1, :]
                        z_memory_similarity_matrix_1 = torch.einsum(
                            "nd,md->nm", z1[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN partition embeddings (for each view)
                        if z_memory_similarity_matrix_1.dim() < 2:
                            _, topk_indices_1 = torch.topk(
                                similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        elif z_memory_similarity_matrix_1.shape[1] <= k:
                            _, topk_indices_1 = torch.topk(
                                similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        else:
                            _, topk_indices_1 = torch.topk(
                                z_memory_similarity_matrix_1, k, dim=1)

                    if indices_2.nelement() == 0:
                        # sanity check that all of the selected clusters more
                        # than 0 assigned memory embeddings
                        _, topk_indices_2 = torch.topk(
                            similarity_matrix2[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding &
                        # selected partition embeddings
                        tmp = bank_normed[indices_2, :].unsqueeze(
                            0) if indices_2.nelement() == 1 else bank_normed[indices_2, :]
                        z_memory_similarity_matrix_2 = torch.einsum(
                            "nd,md->nm", z2[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN partition embeddings (for each view)
                        if z_memory_similarity_matrix_2.dim() < 2:
                            _, topk_indices_2 = torch.topk(
                                similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        elif z_memory_similarity_matrix_2.shape[1] < k:
                            _, topk_indices_2 = torch.topk(
                                similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        else:
                            _, topk_indices_2 = torch.topk(
                                z_memory_similarity_matrix_2, k, dim=1)

                    if topk_indices_1.dim() < 2:
                        topk_indices_1 = topk_indices_1.unsqueeze(0)
                    if topk_indices_2.dim() < 2:
                        topk_indices_2 = topk_indices_2.unsqueeze(0)

                    # concat topk NN embeddings to original embeddings for each view
                    for j in range(k):
                        z1_final = torch.cat((z1_final, torch.index_select(
                            bank, dim=0, index=topk_indices_1[:, j])), 0)
                        z2_final = torch.cat((z2_final, torch.index_select(
                            bank, dim=0, index=topk_indices_2[:, j])), 0)

                # concat the embeddings of the 2 views
                z = torch.cat((z1_final, z2_final), 0)
            else:
                # will only occur if the protoypes are initialized before the
                # memory is full for the first time (in practice not possible)

                # find indices of topk NN memory embeddings for each view
                _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
                _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

                # concat topk NN embeddings to original embeddings for each view
                for i in range(k):
                    z1 = torch.cat((z1, torch.index_select(
                        bank, dim=0, index=topk_indices_1[:, i])), 0)
                    z2 = torch.cat((z2, torch.index_select(
                        bank, dim=0, index=topk_indices_2[:, i])), 0)

                # concat the embeddings of the 2 views
                z = torch.cat((z1, z2), 0)
            return z
        else:
            return output


def clusterer(z, algo='kmeans', n_clusters=5, metric='euclidean', hdb_min_cluster_size=4):
    """
    Performs a clustering step on a set of input embeddings.

    Arguments:
        - z (np.array): the input embeddings to be clustered
        - algo (str): choice of clustering algorithm (optional)
        - n_clusters (int): number clusters to create (optional)
        - metric (str): distance metric in case of hdbscan clustering algorithm
            (optional)
        - hdb_min_cluster_size (int): minimum cluster size for hdbscan

    Returns:
        the clusterer object, the predicted cluster asssignments and the 
        corresponding probabilities
    """
    predicted_labels = None
    probs = None
    if algo == 'kmeans':
        clf = cluster.KMeans(n_clusters=n_clusters, n_init=10)
        predicted_labels = clf.fit_predict(z)
    elif algo == 'hdbscan':
        clf = hdbscan.HDBSCAN(
            metric=metric, min_cluster_size=hdb_min_cluster_size, core_dist_n_jobs=4)
        clf.fit(z)
        predicted_labels = clf.labels_
        probs = clf.probabilities_
    return clf, predicted_labels, probs
