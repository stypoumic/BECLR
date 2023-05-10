import torch
from lightly.loss.memory_bank import MemoryBankModule
from sklearn import cluster
import hdbscan
import torch.nn.functional as F
from utils import visualize_memory_embeddings
import numpy as np
import torch.distributed as dist


class NNmemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16):
        super(NNmemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                epoch: int,
                args,
                k: int = 5,
                labels: torch.Tensor = None,
                update: bool = False):

        # split embeddings of the 2 views
        bsz = output.shape[0] // 2
        z1, z2 = torch.split(
            output, [bsz, bsz], dim=0)

        # Add latest batch to the memory queue (update memory only from 1st view)
        z1, bank = super(NNmemoryBankModule, self).forward(
            z1, labels, update)
        bank = bank.to(output.device).t()

        output = torch.cat((z1, z2), 0)

        # only return the kNN features in case the memory start
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

            # find indices of topk NN for each view
            _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
            _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

            # concat topk NN embeddings for each view
            out1 = torch.index_select(bank, dim=0, index=topk_indices_1[:, 0])
            out2 = torch.index_select(bank, dim=0, index=topk_indices_2[:, 0])
            for i in range(k-1):
                out1 = torch.cat((out1, torch.index_select(
                    bank, dim=0, index=topk_indices_1[:, i])), 0)
                out2 = torch.cat((out2, torch.index_select(
                    bank, dim=0, index=topk_indices_2[:, i])), 0)

            # concat the embeddings of the 2 views
            output = torch.cat((out1, out2), 0)
            return output
        else:
            return output

    def get_top_kNN(self,
                    output: torch.Tensor,
                    epoch: int,
                    args,
                    k: int = 5,
                    labels: torch.Tensor = None,
                    update: bool = False):

        # split embeddings of the 2 views
        bsz = output.shape[0] // 2
        z1, z2 = torch.split(
            output, [bsz, bsz], dim=0)

        # Add latest batch to the memory queue (update memory only from 1st view)
        z1, bank = super(NNmemoryBankModule, self).forward(
            z1, labels, update)
        bank = bank.to(output.device).t()

        output = torch.cat((z1, z2), 0)

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
        clf = hdbscan.HDBSCAN(metric=metric, min_cluster_size=hdb_min_cluster_size, core_dist_n_jobs=4)
        clf.fit(z)
        predicted_labels = clf.labels_
        probs = clf.probabilities_
    return clf, predicted_labels, probs


    

class NNmemoryBankModule2(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16, origin: str = None):
        super(NNmemoryBankModule2, self).__init__(size)
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
        self.topk1 = 1
        self.topk2 = 1
        self.origin = origin
    
    def load_memory_bank(self, bank: torch.Tensor, ptr: torch.long):
        self.bank = bank
        self.ptr = ptr

    @torch.no_grad()
    def cluster_memory_embeddings(self, cluster_algo="kmeans", num_clusters=300, 
                min_cluster_size = 4, rerank=False ):
        bank = self.bank.T.detach()
        
        bank_np = F.normalize(bank.detach()).cpu().numpy()
        
        if cluster_algo == "kmeans":
            clf, labels, _ = clusterer(bank_np,
                                        n_clusters=num_clusters,
                                        algo="kmeans")
            # get cluster means & labels
            centers = clf.cluster_centers_
            self.centers = torch.from_numpy(centers).type_as(bank).cpu()
            self.labels = torch.from_numpy(labels).type_as(bank).long().cpu()
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
            tmp = labels_masked.view(labels_masked.size(0), 1).expand(-1, bank.size(1))

            # (#UN x 512)
            unique_labels, labels_count = tmp.unique(dim=0, return_counts=True)

            # get cluster means
            centers = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, tmp.type(torch.int64), bank)
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
            self.labels[samples_per_cluster*i: samples_per_cluster*(i+1)] = torch.ones(samples_per_cluster) * i

        # shuffle labels and memory
        random_perm = torch.randperm(self.labels.shape[0])
        self.bank = self.bank[:, random_perm]
        self.labels = self.labels[random_perm]



    # @torch.no_grad()
    # def add_memory_embdeddings(self,
    #             z: torch.Tensor, bank, sim_threshold=0.6, topk=3):
    #     centers = self.centers.clone().cuda()
    #     labels = self.labels.clone().cuda()

    #     # Normalize batch & memory embeddings
    #     z_normed = torch.nn.functional.normalize(z, dim=1)
    #     centers = torch.nn.functional.normalize(centers, dim=1)
    #     bank_normed = torch.nn.functional.normalize(bank, dim=0).T

    #     # create similarity matrix between batch embeddings & cluster centers
    #     z_center_similarity_matrix = torch.einsum(
    #         "nd,md->nm", z_normed, centers)

    #     if z_center_similarity_matrix.shape[1] < topk:
    #         topk = z_center_similarity_matrix.shape[1]

    #     # find top3 cluster centers for each batch embedding
    #     _, topk_clusters = torch.topk(z_center_similarity_matrix, topk, dim=1)
    #     #--------------vectorized
    #     for i in range(topk_clusters.shape[0]):
    #         clusters = topk_clusters[i,:]
    #         # find indexes that belong to the topk clusters for batch embedding i
    #         indices = (labels[..., None] == clusters).any(-1).nonzero().squeeze()

    #         # create similarity matrix between batch embedding & selected memory embeddings
    #         z_memory_similarity_matrix = torch.einsum(
    #             "nd,md->nm", z_normed[i,:].unsqueeze(0), bank_normed[indices, :])

    #         # find most similar memory embedding from top3 clusters
    #         sim, top1 = torch.topk(z_memory_similarity_matrix, 1, dim=1)
    #         # replace most similar memory embedding with batch embedding, if 
    #         # their cos similarity is above the threshold
    #         if sim.squeeze().float() > sim_threshold:
    #             index = indices[top1.squeeze()]
    #             self.bank[:, index] = z[i,:].unsqueeze(0)

    
    @torch.no_grad()
    def add_memory_embdeddings_OT(self, args, z: torch.Tensor):
        centers = self.centers.clone().cpu()

        # Normalize batch & memory embeddings
        z_normed = torch.nn.functional.normalize(z, dim=1)  #BS x D
        centers_normed = torch.nn.functional.normalize(centers, dim=1).cuda() #K x D

        # create cost matrix between batch embeddings & cluster centers
        Q = torch.einsum("nd,md->nm", z_normed, centers_normed)    #BS x K

        # apply optimal transport between batch embeddings and cluster centers
        Q = distributed_sinkhorn(Q, args.epsilon, args.sinkhorn_iterations) #BS x K

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
                labels_count = torch.cat((labels_count, torch.tensor([0.001])), 0)
        # _, indices = torch.sort(unique_labels, 0)
        # sort_index = indices[:,0]
        # unique_labels = unique_labels[sort_index, :]
        # labels_count

        # get cluster means
        centers_next = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, view.type(torch.int64), bank).cpu() #UN x 512
        centers_next = centers_next / labels_count.float().unsqueeze(1)

        # for clusters with no assignments, use their center from the perviosu iteration
        for i in deleted_labels:
            centers_next[i, :] = centers[i, :]

        self.centers = centers_next

        return z, bank.T

    def get_top_kNN(self,
                    output: torch.Tensor,
                    epoch: int,
                    args,
                    k: int = 5,
                    labels: torch.Tensor = None,
                    update: bool = False):

        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

        # split embeddings of the 2 views
        bsz = output.shape[0]
        
        if self.start_clustering == False:
            # if memory is full for the first time
            if ptr + bsz >= self.size:
            # if epoch >= args.memory_start_epoch:
                # cluster memory embeddings for the first time
                self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                # self.random_memory_split(args)
                self.start_clustering = True
                print(self.labels.unique(return_counts=True)[-1].size())
                print("--Unique Labels Counts--: {}-------\n".format(self.labels.unique(return_counts=True)[-1]))
                # Visualize memory embeddings using tSNE
                if self.origin == "teacher" or self.origin == "student":
                    visualize_memory_embeddings(np.array(self.bank.T.detach().cpu()),
                                                np.array(self.labels.detach().cpu()), 
                                                args.num_clusters, args.save_path, 
                                                origin=self.origin, epoch=epoch)

                # Add latest batch to the memory queue using Optimal Transport
                output, bank = self.add_memory_embdeddings_OT(args, output)
                use_clustering = True if args.use_cluster_select else False   
            else:
                # Add latest batch to the memory queue (update memory only from both view)
                output, bank = super(NNmemoryBankModule2, self).forward(
                    output, labels, update)
                use_clustering = False
        else:
            if epoch % args.visual_freq == 0 and epoch != self.last_cluster_epoch:
                self.last_cluster_epoch = epoch
                print(self.labels.unique(return_counts=True)[-1].size())
                print("--Unique Labels Counts--: {}-------\n".format(self.labels.unique(return_counts=True)[-1]))
                # Visualize memory embeddings using tSNE
                if self.origin == "teacher" or self.origin == "student":
                    visualize_memory_embeddings(np.array(self.bank.T.detach().cpu()),
                                                np.array(self.labels.detach().cpu()), 
                                                args.num_clusters, args.save_path, 
                                                origin=self.origin, epoch=epoch)
                    
            # Add latest batch to the memory queue using Optimal Transport
            output, bank = self.add_memory_embdeddings_OT(args, output)
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
                _, topk_clusters_1 = torch.topk(z_center_similarity_matrix_1, self.topk1, dim=1)
                _, topk_clusters_2 = torch.topk(z_center_similarity_matrix_2, self.topk2, dim=1)

                z1_final = z1.clone()
                z2_final = z2.clone()
                for i in range(topk_clusters_1.shape[0]):

                    clusters_1 = topk_clusters_1[i,:]
                    clusters_2 = topk_clusters_2[i,:]

                    # find indexes that belong to the top cluster for batch embedding i
                    indices_1 = (labels[..., None] == clusters_1).any(-1).nonzero().squeeze()
                    indices_2 = (labels[..., None] == clusters_2).any(-1).nonzero().squeeze()

                    if indices_1.nelement() == 0:
                        _, topk_indices_1 = torch.topk(similarity_matrix1[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding & selected memory embeddings
                        tmp = bank_normed[indices_1, :].unsqueeze(0) if indices_1.nelement() == 1 else bank_normed[indices_1, :]
                        z_memory_similarity_matrix_1 = torch.einsum(
                            "nd,md->nm", z1[i,:].unsqueeze(0), tmp)
                        
                        # find indices of topk NN for each view
                        if z_memory_similarity_matrix_1.dim() < 2:
                            _, topk_indices_1 = torch.topk(similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        elif z_memory_similarity_matrix_1.shape[1] <= k:
                            _, topk_indices_1 = torch.topk(similarity_matrix1[i, :], k, dim=0)
                            topk_indices_1 = topk_indices_1.unsqueeze(0)
                        else:
                            _, topk_indices_1 = torch.topk(z_memory_similarity_matrix_1, k, dim=1)

                    if indices_2.nelement() == 0:
                        _, topk_indices_2 = torch.topk(similarity_matrix2[i, :], k, dim=0)
                    else:
                        # create similarity matrix between batch embedding & selected memory embeddings
                        tmp = bank_normed[indices_2, :].unsqueeze(0) if indices_2.nelement() == 1 else bank_normed[indices_2, :]
                        z_memory_similarity_matrix_2 = torch.einsum(
                            "nd,md->nm", z2[i,:].unsqueeze(0), tmp)
                        
                        # find indices of topk NN for each view
                        if z_memory_similarity_matrix_2.dim() < 2:
                            _, topk_indices_2 = torch.topk(similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        elif z_memory_similarity_matrix_2.shape[1] < k:
                            _, topk_indices_2 = torch.topk(similarity_matrix2[i, :], k, dim=0)
                            topk_indices_2 = topk_indices_2.unsqueeze(0)
                        else:
                            _, topk_indices_2 = torch.topk(z_memory_similarity_matrix_2, k, dim=1)
                    
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

    # def get_top_kNN(self,
    #                 output: torch.Tensor,
    #                 epoch: int,
    #                 args,
    #                 k: int = 5,
    #                 labels: torch.Tensor = None,
    #                 update: bool = False):

    #     ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

    #     # split embeddings of the 2 views
    #     bsz = output.shape[0] // 2
    #     z1, z2 = torch.split(
    #         output, [bsz, bsz], dim=0)
        
    #     # if memory is full
    #     if ptr + bsz >= self.size:
    #         # cluster memory embeddings for the first time
    #         if self.start_clustering == False:
    #             self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
    #             self.start_clustering = True

    #             # Visualize memory embeddings using tSNE
    #             if self.origin == "teacher" or self.origin == "student":
    #                 visualize_memory_embeddings(np.array(self.bank.T.detach().cpu()),
    #                                             np.array(self.labels.detach().cpu()), 
    #                                             args.num_clusters, args.save_path, 
    #                                             origin=self.origin, epoch=epoch)
            
    #         # cluster memory embeddings every args.cluster_freq epochs
    #         elif self.start_clustering == True and epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
    #             self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
    #             self.last_cluster_epoch = epoch
    #             print("--Clusters Size--: {}-------\n".format(self.centers.size()))
    #             print("--Unique Labels Counts--: {}-------\n".format(self.labels.unique(return_counts=True)))
                
    #             # Visualize memory embeddings using tSNE
    #             if self.origin == "teacher" or self.origin == "student":
    #                 visualize_memory_embeddings(np.array(self.bank.T.detach().cpu()),
    #                                             np.array(self.labels.detach().cpu()), 
    #                                             args.num_clusters, args.save_path, 
    #                                             origin=self.origin, epoch=epoch)

    #         bank = self.bank.clone().cuda().detach()
    #         # Add latest batch to the memory queue based on their most similar memory cluster centers
    #         self.add_memory_embdeddings(z1, bank, sim_threshold=args.sim_threshold)
    #         use_clustering = True

    #     else:
    #         # Add latest batch to the memory queue (update memory only from 1st view)
    #         z1, bank = super(NNmemoryBankModule2, self).forward(
    #             z1, labels, update)
    #         use_clustering = False

    #     output = torch.cat((z1, z2), 0)
    #     bank = bank.to(output.device).t()

    #     # only concat the nearest neighbor features in case the memory start
    #     # epoch has passed
    #     if epoch >= args.memory_start_epoch:
    #         # Normalize batch & memory embeddings
    #         output_normed = torch.nn.functional.normalize(output, dim=1)
    #         bank_normed = torch.nn.functional.normalize(bank, dim=1)

    #         # split embeddings of the 2 views
    #         z1, z2 = torch.split(
    #             output_normed, [args.batch_size, args.batch_size], dim=0)
            
    #         # create similarity matrix between batch & memory embeddings
    #         similarity_matrix1 = torch.einsum(
    #             "nd,md->nm", z1, bank_normed)
    #         similarity_matrix2 = torch.einsum(
    #             "nd,md->nm", z2, bank_normed)
            
    #         # if clustering is used for memory upating, use clustering for NN selection as well
    #         if use_clustering:
    #             centers = self.centers.clone().cuda()
    #             labels = self.labels.clone().cuda()

    #             # Normalize batch & memory embeddings
    #             centers = torch.nn.functional.normalize(centers, dim=1)

    #             # create similarity matrix between batch embeddings & cluster centers
    #             z_center_similarity_matrix_1 = torch.einsum(
    #                 "nd,md->nm", z1, centers)
    #             z_center_similarity_matrix_2 = torch.einsum(
    #                 "nd,md->nm", z2, centers)

    #             if z_center_similarity_matrix_1.shape[1] < self.topk2:
    #                 self.topk1 = z_center_similarity_matrix_1.shape[1]
    #             if z_center_similarity_matrix_2.shape[1] < self.topk1:
    #                 self.topk2 = z_center_similarity_matrix_2.shape[1]

    #             # find top3 cluster centers for each batch embedding
    #             _, topk_clusters_1 = torch.topk(z_center_similarity_matrix_1, self.topk1, dim=1)
    #             _, topk_clusters_2 = torch.topk(z_center_similarity_matrix_2, self.topk2, dim=1)

    #             z1_final = z1.clone()
    #             z2_final = z2.clone()
    #             for i in range(topk_clusters_1.shape[0]):

    #                 clusters_1 = topk_clusters_1[i,:]
    #                 clusters_2 = topk_clusters_2[i,:]

    #                 # find indexes that belong to the topk clusters for batch embedding i
    #                 indices_1 = (labels[..., None] == clusters_1).any(-1).nonzero().squeeze()
    #                 indices_2 = (labels[..., None] == clusters_2).any(-1).nonzero().squeeze()

    #                 # create similarity matrix between batch embedding & selected memory embeddings
    #                 z_memory_similarity_matrix_1 = torch.einsum(
    #                     "nd,md->nm", z1[i,:].unsqueeze(0), bank_normed[indices_1, :])
    #                 z_memory_similarity_matrix_2 = torch.einsum(
    #                     "nd,md->nm", z2[i,:].unsqueeze(0), bank_normed[indices_2, :])

    #                 # find indices of topk NN for each view
    #                 if z_memory_similarity_matrix_1.dim() < 2:
    #                     _, topk_indices_1 = torch.topk(similarity_matrix1[i, :], k, dim=0)
    #                     topk_indices_1 = topk_indices_1.unsqueeze(0)
    #                 elif z_memory_similarity_matrix_1.shape[1] <= k:
    #                     _, topk_indices_1 = torch.topk(similarity_matrix1[i, :], k, dim=0)
    #                     topk_indices_1 = topk_indices_1.unsqueeze(0)
    #                 else:
    #                     _, topk_indices_1 = torch.topk(z_memory_similarity_matrix_1, k, dim=1)

    #                 if z_memory_similarity_matrix_2.dim() < 2:
    #                     _, topk_indices_2 = torch.topk(similarity_matrix2[i, :], k, dim=0)
    #                     topk_indices_2 = topk_indices_2.unsqueeze(0)
    #                 elif z_memory_similarity_matrix_2.shape[1] < k:
    #                     _, topk_indices_2 = torch.topk(similarity_matrix2[i, :], k, dim=0)
    #                     topk_indices_2 = topk_indices_2.unsqueeze(0)
    #                 else:
    #                     _, topk_indices_2 = torch.topk(z_memory_similarity_matrix_2, k, dim=1)
                    
    #                 # concat topk NN embeddings to original embeddings for each view
    #                 for j in range(k):
    #                     z1_final = torch.cat((z1_final, torch.index_select(
    #                         bank, dim=0, index=topk_indices_1[:, j])), 0)
    #                     z2_final = torch.cat((z2_final, torch.index_select(
    #                         bank, dim=0, index=topk_indices_2[:, j])), 0)
                        
    #             # concat the embeddings of the 2 views
    #             z = torch.cat((z1_final, z2_final), 0)
    #         else:
    #             # find indices of topk NN for each view
    #             _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
    #             _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

    #             # concat topk NN embeddings to original embeddings for each view
    #             for i in range(k):
    #                 z1 = torch.cat((z1, torch.index_select(
    #                     bank, dim=0, index=topk_indices_1[:, i])), 0)
    #                 z2 = torch.cat((z2, torch.index_select(
    #                     bank, dim=0, index=topk_indices_2[:, i])), 0)

    #             # concat the embeddings of the 2 views
    #             z = torch.cat((z1, z2), 0)
    #         return z
    #     else:
    #         return output
        
    # def forward(self,
    #             output: torch.Tensor,
    #             epoch: int,
    #             args,
    #             k: int = 5,
    #             labels: torch.Tensor = None,
    #             update: bool = False):
        
    #     ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

    #     # split embeddings of the 2 views
    #     bsz = output.shape[0] // 2
    #     z1, z2 = torch.split(
    #         output, [bsz, bsz], dim=0)
        
    #     # if memory is full
    #     if ptr + bsz >= self.size:
    #         # cluster memory embeddings for the first time
    #         if self.start_clustering == False:
    #             self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
    #             self.start_clustering = True
            
    #         # cluster memory embeddings every args.cluster_freq epochs
    #         elif epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
    #             self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
    #             self.last_cluster_epoch = epoch

    #         bank = self.bank.clone().cuda().detach()
    #         # Add latest batch to the memory queue based on their most similar memory cluster centers
    #         self.add_memory_embdeddings(z1, bank, sim_threshold=args.sim_threshold)
           
    #     else:
    #         # Add latest batch to the memory queue (update memory only from 1st view)
    #         z1, bank = super(NNmemoryBankModule2, self).forward(
    #             z1, labels, update)
            
    #     bank = bank.to(output.device).t()
    #     output = torch.cat((z1, z2), 0)

    #     # only return the kNN features in case the memory start
    #     # epoch has passed
    #     if epoch >= args.memory_start_epoch:
    #         # Normalize batch & memory embeddings
    #         output_normed = torch.nn.functional.normalize(output, dim=1)
    #         bank_normed = torch.nn.functional.normalize(bank, dim=1)

    #         # split embeddings of the 2 views
    #         z1, z2 = torch.split(
    #             output_normed, [args.batch_size, args.batch_size], dim=0)
            
    #         # create similarity matrix between batch & memory embeddings
    #         similarity_matrix1 = torch.einsum(
    #             "nd,md->nm", z1, bank_normed)
    #         similarity_matrix2 = torch.einsum(
    #             "nd,md->nm", z2, bank_normed)

    #         # find indices of topk NN for each view
    #         _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
    #         _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

    #         # concat topk NN embeddings for each view
    #         out1 = torch.index_select(bank, dim=0, index=topk_indices_1[:, 0])
    #         out2 = torch.index_select(bank, dim=0, index=topk_indices_2[:, 0])
    #         for i in range(k-1):
    #             out1 = torch.cat((out1, torch.index_select(
    #                 bank, dim=0, index=topk_indices_1[:, i])), 0)
    #             out2 = torch.cat((out2, torch.index_select(
    #                 bank, dim=0, index=topk_indices_2[:, i])), 0)

    #         # concat the embeddings of the 2 views
    #         output = torch.cat((out1, out2), 0)
    #         return output
    #     else:
    #         return output

@torch.no_grad()
def distributed_sinkhorn(out, epsilon, iterations):
    # Q is K-by-B for consistency with notations from our paper
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]   # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

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
    # print(torch.sum(Q, dim=1, keepdim=True))
    return Q.t()


        