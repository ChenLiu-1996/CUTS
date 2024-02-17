import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Need to provide the `anchor, positive` patch pairs within the same image.
    Negative sample can be directly inferred by using
    the positive sample from a different anchor in the same image.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

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

            assert Z_anchors.shape == Z_pos.shape

            z1 = torch.nn.functional.normalize(Z_anchors, p=2, dim=1)
            z2 = torch.nn.functional.normalize(Z_pos, p=2, dim=1)
            z = torch.cat((z1, z2), dim=0)

            # Compute similarity matrix
            # Note that we refactor the `exp` and `1/temperature` operations here.
            sim_matrix = torch.exp(torch.matmul(z, z.T) / self.temperature)

            # Masks to identify positive and all valid negatives for each example
            positive_mask = torch.cat((
                torch.cat((torch.zeros((S, S), dtype=torch.bool), torch.eye(S, dtype=torch.bool)), dim=0),
                torch.cat((torch.eye(S, dtype=torch.bool), torch.zeros((S, S), dtype=torch.bool)), dim=0),
                                    ), dim=1)
            negative_mask = torch.cat((
                torch.cat((~torch.eye(S, dtype=torch.bool), ~torch.eye(S, dtype=torch.bool)), dim=0),
                torch.cat((~torch.eye(S, dtype=torch.bool), ~torch.eye(S, dtype=torch.bool)), dim=0),
                                    ), dim=1)

            # Selecting the positive examples for each anchor
            score_pos = sim_matrix[positive_mask].view(2 * S, 1)

            # Sum of all similarities for negative pairs
            score_neg = sim_matrix[negative_mask].view(2 * S, -1).sum(dim=1, keepdim=True)

            # Calculating the InfoNCE loss as the log ratio
            loss += -torch.log(score_pos / (score_pos + score_neg))

        # Average loss over the batch
        return loss.mean() / B
