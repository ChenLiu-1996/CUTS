import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Need to provide the `anchor, positive` patch pairs within the same image.
    Negative sample can be directly inferred by using
    the positive sample from a different anchor in the same image.
    """

    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-7

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor):
        """
        Assuming `anchors` and `positives` to have dimension [B, S, L]
            B: batch size
            S: number of sampled patches per image
            L: latent vector dimension
        """
        assert len(anchors.shape) == 3
        assert anchors.shape == positives.shape

        B, S, _ = anchors.shape

        loss = 0
        # We would like to learn contrastively across patches in the same image.
        # So we will use all sampled patches within the same batch idx to compute the loss.
        for batch_idx in range(B):
            Z_anchors = anchors[batch_idx, ...]
            Z_pos = positives[batch_idx, ...]

            # Create a matrix that represent the [i,j] entries of positive pairs.
            pos_pair_ij = torch.diag(torch.ones(S)).bool()

            Z_anchor = F.normalize(input=Z_anchors, p=2, dim=-1)
            Z_pos = F.normalize(input=Z_pos, p=2, dim=-1)
            sim_matrix = torch.matmul(Z_anchor, Z_pos.T)

            # Entries noted by 1's in `pos_pair_ij` are similarities of positive pairs.
            numerator = torch.sum(
                torch.exp(sim_matrix[pos_pair_ij] / self.temperature))

            # Entries elsewhere are similarities of negative pairs.
            denominator = torch.sum(
                torch.exp(sim_matrix[~pos_pair_ij] / self.temperature))

            loss += -torch.log(numerator /
                               (denominator + self.epsilon) + self.epsilon)

        return loss / B
