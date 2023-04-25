import torch
from lightly.loss.memory_bank import MemoryBankModule


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
        



from sklearn import cluster
import hdbscan
import torch.nn.functional as F

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
    def __init__(self, size: int = 2 ** 16):
        super(NNmemoryBankModule2, self).__init__(size)
        self.clusters = None
        self.labels = None
        self.start_clustering = False
        self.last_cluster_epoch = 0
    
    def cluster_memory_embeddings(self, cluster_algo="kmeans", num_clusters=300, 
                min_cluster_size = 4, rerank=False ):
        bank = self.bank.T
        
        bank_np = F.normalize(bank.detach()).cpu().numpy()
        
        if cluster_algo == "kmeans":
            clf, labels, _ = clusterer(bank_np,
                                        n_clusters=num_clusters,
                                        algo="kmeans")
            # get cluster means & labels
            centers = clf.cluster_centers_
            self.centers = torch.from_numpy(centers).type_as(bank)
            self.labels = torch.from_numpy(labels).type_as(bank).long()
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
            self.centers = centers / labels_count.float().unsqueeze(1)
            self.labels = labels 


    
    def add_memory_embdeddings(self,
                z: torch.Tensor, sim_threshold=0.6, topk=3):
        bank = self.bank.clone().detach().cuda()
        centers = self.centers.clone().detach().cuda()
        labels = self.labels.clone().detach().cuda()

        # Normalize batch & memory embeddings
        z_normed = torch.nn.functional.normalize(z, dim=1)
        canters_normed = torch.nn.functional.normalize(centers, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=0).T

        # create similarity matrix between batch embeddings & cluster centers
        z_center_similarity_matrix = torch.einsum(
            "nd,md->nm", z_normed, canters_normed)

        if z_center_similarity_matrix.shape[1] < topk:
            topk = z_center_similarity_matrix.shape[1]

        # find top3 cluster centers for each batch embedding
        _, topk_clusters = torch.topk(z_center_similarity_matrix, topk, dim=1) 
        #--------------vectorized
        for i in range(topk_clusters.shape[0]):
            clusters = topk_clusters[i,:]
            # find indexes that belong to the topk clusters for batch embedding i
            indices = (labels[..., None] == clusters).any(-1).nonzero().squeeze()

            # create similarity matrix between batch embedding & selected memory embeddings
            z_memory_similarity_matrix = torch.einsum(
                "nd,md->nm", z_normed[i,:].unsqueeze(0), bank_normed[indices, :])

            # find most similar memory embedding from top3 clusters
            sim, top1 = torch.topk(z_memory_similarity_matrix, 1, dim=1)
            # replace most similar memory embedding with batch embedding, if 
            # their cos similarity is above the threshold
            if sim.squeeze().float() > sim_threshold:
                index = indices[top1.squeeze()]
                self.bank[:, index] = z[i,:].unsqueeze(0)

        return z, bank

    def forward(self,
                output: torch.Tensor,
                epoch: int,
                args,
                k: int = 5,
                labels: torch.Tensor = None,
                update: bool = False):
        
        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

        # split embeddings of the 2 views
        bsz = output.shape[0] // 2
        z1, z2 = torch.split(
            output, [bsz, bsz], dim=0)
        
        # if memory is full
        if ptr + bsz >= self.size:
            # cluster memory embeddings for the first time
            if self.start_clustering == False:
                self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.start_clustering = True
            
            # cluster memory embeddings every args.cluster_freq epochs
            elif epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
                self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            # Add latest batch to the memory queue based on their most similar memory cluster centers
            z1 , bank = self.add_memory_embdeddings(z1, sim_threshold=args.sim_threshold)

        else:
            # Add latest batch to the memory queue (update memory only from 1st view)
            z1, bank = super(NNmemoryBankModule2, self).forward(
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

        ptr = int(self.bank_ptr) if self.bank.nelement() != 0 else 0

        # split embeddings of the 2 views
        bsz = output.shape[0] // 2
        z1, z2 = torch.split(
            output, [bsz, bsz], dim=0)
        
        # if memory is full
        if ptr + bsz >= self.size:
            # cluster memory embeddings for the first time
            if self.start_clustering == False:
                self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.start_clustering = True
            
            # cluster memory embeddings every args.cluster_freq epochs
            elif self.start_clustering == True and epoch % args.cluster_freq == 0 and epoch != self.last_cluster_epoch:
                self.cluster_memory_embeddings(cluster_algo=args.cluster_algo, num_clusters=args.num_clusters)
                self.last_cluster_epoch = epoch

            # Add latest batch to the memory queue based on their most similar memory cluster centers
            z1 , bank = self.add_memory_embdeddings(z1, sim_threshold=args.sim_threshold)

        else:
            # Add latest batch to the memory queue (update memory only from 1st view)
            z1, bank = super(NNmemoryBankModule2, self).forward(
                z1, labels, update)

        output = torch.cat((z1, z2), 0)
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




        