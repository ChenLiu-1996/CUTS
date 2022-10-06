import random
from typing import Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


class PatchSampler(object):
    """
    `PatchSampler` samples patches of a [B, C, H, W] image to form
    positive and negative pairs for contrastive learning purposes.

    Current strategy is to sample N anchor patches,
    then sample 1 positive patch and 1 negative patch for each anchor patch.
    """

    def __init__(self, random_seed: int = None, patch_size: int = 7, pairs_per_batch: int = 1):
        self.random_seed = random_seed
        self.patch_size = patch_size
        self.pairs_per_batch = pairs_per_batch

    def sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        `image` dimension: [B, C, H, W]
        Dimension of returned 3 Tensors: [B, pairs_per_batch, 2]
        """

        B, _, H, W = image.shape

        # last dim is 2, in the (x, y) format.
        anchors_xy = np.zeros((B, self.pairs_per_batch, 2), dtype=int)
        positives_xy = np.zeros_like(anchors_xy)
        negatives_xy = np.zeros_like(anchors_xy)

        x_range = (self.patch_size // 2, H - self.patch_size // 2)
        y_range = (self.patch_size // 2, W - self.patch_size // 2)

        # First sample `pairs_per_batch` anchors. Then find pos/neg patches against each of them.
        random.seed(self.random_seed)
        for batch_idx in range(B):
            for patch_idx in range(self.pairs_per_batch):
                anchors_xy[batch_idx, patch_idx, :] = [
                    random.randrange(start=x_range[0], stop=x_range[1]),
                    random.randrange(start=y_range[0], stop=y_range[1]),
                ]

                # Sample nearby the anchor and check SSIM. Repeat if not similar enough.
                pos_xy_candidate = sample_xy_nearby(
                    anchors_xy[batch_idx, patch_idx, :], H=H, W=W)
                while not similar_enough(image[batch_idx, ...].cpu().detach().numpy(),
                                         x1y1=anchors_xy[batch_idx,
                                                         patch_idx, :],
                                         x2y2=pos_xy_candidate, patch_size=self.patch_size):
                    pos_xy_candidate = sample_xy_nearby(
                        anchors_xy[batch_idx, patch_idx, :], H=H, W=W)
                positives_xy[batch_idx, patch_idx, :] = pos_xy_candidate

                # Randomly sample a patch, and it will be a negative patch.
                negatives_xy[batch_idx, patch_idx, :] = [
                    random.randrange(start=x_range[0], stop=x_range[1]),
                    random.randrange(start=y_range[0], stop=y_range[1]),
                ]

        assert anchors_xy.shape == positives_xy.shape and anchors_xy.shape == negatives_xy.shape
        return anchors_xy, positives_xy, negatives_xy


def sample_xy_nearby(xy: Tuple[int, int], H: int, W: int, neighborhood: int = 5) -> Tuple[int, int]:
    return (random.randrange(start=max(xy[0]-neighborhood, 0),
                             stop=min(xy[0]+neighborhood, H)),
            random.randrange(start=max(xy[1]-neighborhood, 0),
                             stop=min(xy[1]+neighborhood, W)))


def similar_enough(image: np.array,
                   x1y1: Tuple[int, int], x2y2: Tuple[int, int],
                   patch_size: int, ssim_thr: float = 0.8) -> bool:
    """
    `image` dimension: [C, H, W]
    """
    patch1 = image[:,
                   x1y1[0]-patch_size//2: x1y1[0]-patch_size//2+patch_size,
                   x1y1[1]-patch_size//2: x1y1[1]-patch_size//2+patch_size]
    patch2 = image[:,
                   x2y2[0]-patch_size//2: x2y2[0]-patch_size//2+patch_size,
                   x2y2[1]-patch_size//2: x2y2[1]-patch_size//2+patch_size]

    # Channel first to channel last to accommodate SSIM.
    patch1 = np.moveaxis(patch1, 0, -1)
    patch2 = np.moveaxis(patch2, 0, -1)
    return ssim(patch1, patch2, multichannel=True) >= ssim_thr
