import numpy as np
import phate
import torch

from .one_shot import OneShotClusterEstimator


class LatentEvaluator(object):
    """
    Evaluate the latent representation of the unsupervised segmentation model.

    Currently implemented:
        Segmentation dice coefficient.

    init @params:
        `oneshot_prior`:
            == 'point': use a center point in one mask to estimate the desired cluster.
            == 'mask': use the entirety of one mask to estimate the desired cluster.
    """

    def __init__(self, oneshot_prior: str = 'point') -> None:
        self.oneshot_estimator = OneShotClusterEstimator(oneshot_prior)

    def dice(self, latent: torch.Tensor, label: torch.Tensor) -> float:
        """
        @params:
        `latent`: latent encoding by the model.
        `label`: ground truth segmentation map.

        `latent` and `label` are expected to have dimension [B, H, W, C].
        """
        ncluster = 6

        latent = latent.cpu().detach().numpy()
        latent = np.moveaxis(latent, 1, -1)  # channel-first to channel-last
        label = label.cpu().detach().numpy()

        B, H, W, _ = latent.shape

        dice_coeffs = []
        for batch_idx in range(B):
            # [H, W, C] to [H x W, C]
            feature = latent[batch_idx]
            feature = feature.reshape((-1, feature.shape[-1]))

            # Perform PHATE clustering.
            phate_operator = phate.PHATE(
                n_components=3, knn=100, n_landmark=500, t=2, verbose=False)
            _phate_embedding = phate_operator.fit_transform(feature)
            clusters = phate.cluster.kmeans(
                phate_operator, n_clusters=ncluster)

            assert clusters.shape == (H * W,)
            clusters = clusters.reshape((H, W))

            # Find the centroid of ground truth label.
            cluster_id = self.oneshot_estimator.estimate_cluster(
                label[batch_idx], clusters)

            seg_pred = clusters == cluster_id
            seg_true = label[batch_idx] > 0

            dice_coeffs.append(dice_coeff(seg_pred, seg_true))

        return dice_coeffs


def dice_coeff(pred: np.array, label: np.array) -> float:
    intersection = np.logical_and(pred, label).sum()
    union = pred.sum() + label.sum()
    dice = (2 * intersection) / union
    return dice
