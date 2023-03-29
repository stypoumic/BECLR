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

        # Add latest batch to the memory queue
        output, bank = super(NNmemoryBankModule, self).forward(
            output, labels, update)
        bank = bank.to(output.device).t()

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

        # only return the nearest neighbor features instead of the originals,
        # in case the NNCLR start epoch has passed
        if epoch > start_epoch:
            return nearest_neighbours
        else:
            return output
