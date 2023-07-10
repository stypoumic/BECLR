import hdbscan
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightly.loss.memory_bank import MemoryBankModule
from sklearn import cluster
from torchmetrics.functional import pairwise_euclidean_distance
from sinkhorn import distributed_sinkhorn, sinkhorn2

from visualize import visualize_memory, visualize_memory_batch


def clusterer(z, algo='kmeans', n_clusters=5, metric='euclidean', hdb_min_cluster_size=4):
    """
    Clusters the points
    :param z: The reduced dataset or the distances
    :param algo: kmeans or hdbscan
    :param n_clusters: n_clusters for kmeans
    :param hdb_min_cluster_size: hdbscan min cluster size
    :param metric: applies only to hdbscan
    :return:
    """
    predicted_labels = None
    probs = None
    if algo == 'kmeans':
        clf = cluster.KMeans(n_clusters=n_clusters, n_init=10)
        predicted_labels = clf.fit_predict(z)
    elif algo == 'hdbscan':
        # if cuml_details is not None:
        #     clf = hdbscan.HDBSCAN(metric=metric, min_cluster_size=hdb_min_cluster_size)
        # else:
        clf = hdbscan.HDBSCAN(
            metric=metric, min_cluster_size=hdb_min_cluster_size, core_dist_n_jobs=4)
        clf.fit(z)
        predicted_labels = clf.labels_
        probs = clf.probabilities_
    return clf, predicted_labels, probs


class NNmemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16, origin: str = None):
        super(NNmemoryBankModule, self).__init__(size)
        # register_buffer => Tensor which is not a parameter, but should be part
        #  of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict
        # (e.g. when we save the model)
        self.register_buffer(
            "centers", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "labels", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )
        self.start_clustering = False
        self.last_cluster_epoch = 0
        self.last_vis_epoch = 0
        self.topk1 = 1
        self.topk2 = 1
        self.origin = origin
        # self.idx = 0            # TO REMOVE

    def load_memory_bank(self, memory_bank: tuple):
        self.bank = memory_bank[0]
        self.bank_ptr = memory_bank[1]
        self.labels = memory_bank[2]
        self.centers = memory_bank[3]
        self.start_clustering = True

    @torch.no_grad()
    def cluster_memory_embeddings(self, cluster_algo="kmeans", num_clusters=300,
                                  min_cluster_size=4, rerank=False):
        bank = self.bank.T.detach()

        bank_np = F.normalize(bank.detach()).cpu().numpy()

        if cluster_algo == "kmeans":
            clf, labels, _ = clusterer(bank_np,
                                       n_clusters=num_clusters,
                                       algo="kmeans")
            # get cluster means & labels
            centers = clf.cluster_centers_
            centers = torch.from_numpy(centers).type_as(bank).cpu()
            labels = torch.from_numpy(labels).type_as(bank).long().cpu()
            # do not upddate the clusters in the memory in case of redundancy from kmeans
            if self.start_clustering and len(labels.unique(return_counts=True)[-1]) < num_clusters:
                return
            self.centers = centers
            self.labels = labels
        else:
            # possibly reranking before clustering
            clf, labels, probs = clusterer(
                bank_np.astype(float),
                algo="hdbscan",
                metric="euclidean" if not rerank else "precomputed",
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

            # (#UN x 512)
            unique_labels, labels_count = tmp.unique(dim=0, return_counts=True)

            # get cluster means
            centers = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
                0, tmp.type(torch.int64), bank)
            self.centers = centers / labels_count.float().unsqueeze(1).cpu()
            self.labels = labels.cpu()

    @torch.no_grad()
    def random_memory_split(self, args):
        bank = self.bank.T.detach()
        # shuffle memory, so that cluster assignments are random
        bank = bank[torch.randperm(bank.shape[0]), :]
        self.centers = torch.zeros(args.num_clusters, bank.shape[1])
        self.labels = torch.zeros(bank.shape[0])

        samples_per_cluster = bank.shape[0] // args.num_clusters

        for i in range(0, args.num_clusters):
            samples = bank[samples_per_cluster*i: samples_per_cluster*(i+1), :]

            self.centers[i, :] = torch.mean(samples, 0, True)
            self.labels[samples_per_cluster*i: samples_per_cluster *
                        (i+1)] = torch.ones(samples_per_cluster) * i

        # shuffle labels and memory
        random_perm = torch.randperm(self.labels.shape[0])
        self.bank = self.bank[:, random_perm]
        self.labels = self.labels[random_perm]

    @torch.no_grad()
    def add_memory_embdeddings_OT(self, args, z: torch.Tensor, dist_metric="cosine", momentum=0.9):
        centers = self.centers.clone().cpu()

        if dist_metric == "cosine":
            # Normalize batch & memory embeddings
            z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
            centers_normed = torch.nn.functional.normalize(
                centers, dim=1).cuda()  # K x D

            # create cost matrix between batch embeddings & cluster centers
            Q = torch.einsum("nd,md->nm", z_normed, centers_normed)  # BS x K
        else:
            # Normalize batch & memory embeddings
            z_normed = torch.nn.functional.normalize(z, dim=1)  # BS x D
            centers_normed = torch.nn.functional.normalize(
                centers, dim=1).cuda()  # K x D
            # create cost matrix between batch embeddings & cluster centers
            Q = pairwise_euclidean_distance(z_normed, centers_normed)

            # create cost matrix between batch embeddings & cluster centers
            # Q = pairwise_euclidean_distance(z, centers.cuda())

        # apply optimal transport between batch embeddings and cluster centers
        if args.use_sinnkhorn_ver2:
            Q = sinkhorn2(z_normed, centers_normed, eps=args.epsilon)
        else:
            Q = distributed_sinkhorn(
                Q, args.epsilon, args.sinkhorn_iterations)  # BS x K

        # get assignments (batch labels)
        batch_labels = torch.argmax(Q, dim=1)

        # add equipartitioned batch to memory (FIFO)
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

        # Update cluster centers
        labels = self.labels.clone().cpu()
        bank = self.bank.clone().cpu().T.detach()

        view = labels.view(labels.size(0), 1).expand(-1, bank.size(1))
        unique_labels, labels_count = view.unique(dim=0, return_counts=True)
        deleted_labels = []
        for i in range(0, centers.shape[0]):
            if i not in unique_labels[:, 0]:
                deleted_labels.append(i)
                label = torch.tensor([[i]]).expand(-1, bank.size(1))
                unique_labels = torch.cat((unique_labels, label), 0)
                labels_count = torch.cat(
                    (labels_count, torch.tensor([0.001])), 0)

        # get cluster means
        centers_next = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
            0, view.type(torch.int64), bank).cpu()  # UN x 512
        centers_next = centers_next / labels_count.float().unsqueeze(1)

        # for clusters with no assignments, use their center from the previous iteration
        for i in deleted_labels:
            centers_next[i, :] = centers[i, :]

        # EMA update of cluster centers
        self.centers = centers_next * (1 - momentum) + momentum * centers

        return z, bank.T

    def get_top_kNN(self,
                    output: torch.Tensor,
                    epoch: int,
                    args,
                    k: int = 5,
                    test_batch_labels: torch.Tensor = None,     # TO REMOVE
                    update: bool = False):

        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

        bsz = output.shape[0]

        if self.start_clustering == False:
            # if memory is full for the first time
            if ptr + bsz >= self.size:
                # cluster memory embeddings for the first time
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)

                self.last_cluster_epoch = epoch
                self.start_clustering = True
                print("--Unique Labels Counts--: {}-------\n".format(
                    self.labels.unique(return_counts=True)[-1]))

                # Add latest batch to the memory queue using Optimal Transport
                output, bank = self.add_memory_embdeddings_OT(
                    args, output, dist_metric=args.memory_dist_metric, momentum=args.memory_momentum)
                use_clustering = True if args.use_cluster_select else False
            else:
                # Add latest batch to the memory queue (update memory only from both view)
                output, bank = super(NNmemoryBankModule, self).forward(
                    output, None, update)
                use_clustering = False
        else:
            if args.recluster and epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
                # restart the memory clusters
                print("--Unique Labels Counts--: {}-------\n".format(
                    self.labels.unique(return_counts=True)[-1]))
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            if len(self.labels.unique(return_counts=True)[-1]) <= 1:
                # restart memory clusters in case the memory embeddings have
                # converged to a single cluster
                # (In practice: not used, but covers the case when not suitable
                # hyperparameters for the OT memory updating have been chosen)
                self.cluster_memory_embeddings(
                    cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            if epoch % args.visual_freq == 0 and epoch != self.last_vis_epoch:
                self.last_vis_epoch = epoch
                # Visualize memory embeddings using UMAP
                if self.origin == "teacher" or self.origin == "student":
                    visualize_memory(self, args.save_path,
                                     self.origin, epoch=epoch, n_samples=args.memory_scale)

            # Add latest batch to the memory queue using Optimal Transport
            output, bank = self.add_memory_embdeddings_OT(
                args, output, dist_metric=args.memory_dist_metric, momentum=args.memory_momentum)
            use_clustering = True if args.use_cluster_select else False

        bank = bank.to(output.device).t()

        # only concat the nearest neighbor features in case the memory start
        # epoch has passed
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

            # if clustering is used for memory upating, use clustering for NN selection as well
            if use_clustering:
                centers = self.centers.clone().cuda()
                labels = self.labels.clone().cuda()

                # Normalize batch & memory embeddings
                centers = torch.nn.functional.normalize(centers, dim=1)

                # create similarity matrix between batch embeddings & cluster centers
                z_center_similarity_matrix_1 = torch.einsum(
                    "nd,md->nm", z1, centers)
                z_center_similarity_matrix_2 = torch.einsum(
                    "nd,md->nm", z2, centers)

                # only in case of hdbscan
                # if z_center_similarity_matrix_1.shape[1] < self.topk2:
                #     self.topk1 = z_center_similarity_matrix_1.shape[1]
                # if z_center_similarity_matrix_2.shape[1] < self.topk1:
                #     self.topk2 = z_center_similarity_matrix_2.shape[1]

                # find top cluster centers for each batch embedding
                _, topk_clusters_1 = torch.topk(
                    z_center_similarity_matrix_1, self.topk1, dim=1)
                _, topk_clusters_2 = torch.topk(
                    z_center_similarity_matrix_2, self.topk2, dim=1)

                ##############################################################################################################
                # # # # TO REMOVE
                # if self.origin == "teacher" or self.origin == "student":
                #     visualize_memory_batch(
                #         z1, test_batch_labels, centers, "C:/GitHub/msiam/visualizations", proj="umap", origin=self.origin, idx=self.idx)
                #     self.idx += 1
                #############################################################################################################

                z1_final = z1.clone()
                z2_final = z2.clone()
                for i in range(topk_clusters_1.shape[0]):

                    clusters_1 = topk_clusters_1[i, :]
                    clusters_2 = topk_clusters_2[i, :]

                    # find indexes that belong to the top cluster for batch embedding i
                    indices_1 = (labels[..., None] ==
                                 clusters_1).any(-1).nonzero().squeeze()
                    indices_2 = (labels[..., None] ==
                                 clusters_2).any(-1).nonzero().squeeze()

                    if indices_1.nelement() == 0:
                        _, topk_indices_1 = torch.topk(
                            similarity_matrix1[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding & selected memory embeddings
                        tmp = bank_normed[indices_1, :].unsqueeze(
                            0) if indices_1.nelement() == 1 else bank_normed[indices_1, :]
                        z_memory_similarity_matrix_1 = torch.einsum(
                            "nd,md->nm", z1[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN for each view
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
                        _, topk_indices_2 = torch.topk(
                            similarity_matrix2[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding & selected memory embeddings
                        tmp = bank_normed[indices_2, :].unsqueeze(
                            0) if indices_2.nelement() == 1 else bank_normed[indices_2, :]
                        z_memory_similarity_matrix_2 = torch.einsum(
                            "nd,md->nm", z2[i, :].unsqueeze(0), tmp)

                        # find indices of topk NN for each view
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
                    #######################
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
                # find indices of topk NN for each view
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
