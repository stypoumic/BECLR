import torch
from lightly.loss.memory_bank import MemoryBankModule


class NNmemoryBankModule(MemoryBankModule):
    def __init__(self, size: int = 2 ** 16):
        super(NNmemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                epoch: int,
                start_epoch: int,
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

        # only return the nearest neighbor features instead of the originals,
        # in case the NNCLR start epoch has passed
        if epoch >= start_epoch:
            # Normalize batch & memory embeddings
            output_normed = torch.nn.functional.normalize(output, dim=1)
            bank_normed = torch.nn.functional.normalize(bank, dim=1)

            # create similarity matrix between batch & memory embeddings
            similarity_matrix = torch.einsum(
                "nd,md->nm", output_normed, bank_normed)

            # find nearest-neighbor for each batch embedding
            index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
            nearest_neighbours = torch.index_select(
                bank, dim=0, index=index_nearest_neighbours)

            return nearest_neighbours
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

            # find indices of top5 NN for each view
            _, topk_indices_1 = torch.topk(similarity_matrix1, k, dim=1)
            _, topk_indices_2 = torch.topk(similarity_matrix2, k, dim=1)

            # concat top5 NN embeddings to original embeddings for each view
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
