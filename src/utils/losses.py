import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss  # For re-export.


class NTXentLoss(nn.Module):
    """
    Need to provide the `anchor, positive` pairs.
    Negative sample can be directly inferred by using
    the positive sample from a different anchor in the same batch.

    TODO: This method is batch-size-dependent.
    Also, it is quite catastrophic when hitting a batch size of 1,
    including when it is a trailing batch.
    Current workaround is to load the entire dataset as a single batch for val and test.
    """

    def __init__(self, temperature: float = 0.5, batch_size: int = 2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-7
        self.batch_size = batch_size

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor):
        assert len(anchors.shape) == 2
        assert anchors.shape == positives.shape

        B = anchors.shape[0]

        # Create a matrix that represent the [i,j] entries of positive pairs.
        pos_pair_ij = torch.diag(torch.ones(B)).bool()

        Z_anchor = F.normalize(input=anchors, p=2, dim=1)
        Z_pos = F.normalize(input=positives, p=2, dim=1)
        sim_matrix = torch.matmul(Z_anchor, Z_pos.T)

        # Entries noted by 1's in `pos_pair_ij` are the positive pairs.
        numerator = torch.sum(
            torch.exp(sim_matrix[pos_pair_ij] / self.temperature))

        # Entries elsewhere are the negative pairs.
        denominator = torch.sum(
            torch.exp(sim_matrix[~pos_pair_ij] / self.temperature))

        loss = -torch.log(numerator / (denominator +
                          self.epsilon) + self.epsilon)

        return loss
