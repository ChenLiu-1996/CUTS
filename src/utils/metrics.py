from typing import Tuple

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
            == 'point': use a center point in each mask to estimate the desired cluster.
    """

    def __init__(self, oneshot_prior: str = 'point', random_seed: int = None) -> None:
        self.oneshot_estimator = OneShotClusterEstimator(oneshot_prior)
        self.random_seed = random_seed

    def dice(self, latent: torch.Tensor, label: torch.Tensor,
             return_additional_info: bool = False) -> Tuple[float, np.array, np.array]:
        """
        @params:
        `latent`: latent encoding by the model.
        `label`: ground truth segmentation map.

        `latent` and `label` are expected to have dimension [B, H, W, C].
        """
        ncluster = 10
        latent = latent.cpu().detach().numpy()
        latent = np.moveaxis(latent, 1, -1)  # channel-first to channel-last
        label = label.cpu().detach().numpy()

        B, H, W, C = latent.shape

        dice_coeffs, multiclass_segs, binary_segs = [], None, None

        for batch_idx in range(B):
            feature_map = latent[batch_idx]
            # [H, W, C] to [H x W, C]
            feature_map = feature_map.reshape((H*W, C))

            # Perform PHATE clustering.
            phate_operator = phate.PHATE(
                n_components=3, knn=100, n_landmark=500, t=2, verbose=False, random_state=self.random_seed)
            phate_operator.fit_transform(feature_map)
            clusters = phate.cluster.kmeans(
                phate_operator, n_clusters=ncluster, random_state=self.random_seed)

            assert clusters.shape == (H * W,)
            clusters = clusters.reshape((H, W))

            # Find the centroid of ground truth label.
            cluster_id = self.oneshot_estimator.estimate_cluster(
                label[batch_idx], clusters)

            seg_pred = clusters == cluster_id
            seg_true = label[batch_idx] > 0

            dice_coeffs.append(dice_coeff(seg_pred, seg_true))

            if multiclass_segs is None:
                multiclass_segs = clusters[np.newaxis, ...]
                binary_segs = seg_pred[np.newaxis, ...]
            else:
                multiclass_segs = np.concatenate(
                    (multiclass_segs, clusters[np.newaxis, ...]), axis=0)
                binary_segs = np.concatenate(
                    (binary_segs, seg_pred[np.newaxis, ...]), axis=0)

        if return_additional_info:
            return dice_coeffs, multiclass_segs, binary_segs, latent
        else:
            return dice_coeffs


def dice_coeff(pred: np.array, label: np.array) -> float:
    intersection = np.logical_and(pred, label).sum()
    dice = (2 * intersection) / (pred.sum() + label.sum())
    return dice
