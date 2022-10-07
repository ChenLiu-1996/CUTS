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

    def __init__(self, random_seed: int = None, patch_size: int = 7, samples_per_batch: int = 1):
        self.random_seed = random_seed
        self.patch_size = patch_size
        self.samples_per_batch = samples_per_batch

    def sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        `image` dimension: [B, C, H, W]
        Dimension of returned Tensors: [B, samples_per_batch, 2]
        The last dimension is 2, in the (h, w) format.
        (h, w) denotes the pixel location -- not using (x, y) to avoid ambiguity against (input, GT).
        """

        B, _, H, W = image.shape

        anchors_hw = np.zeros((B, self.samples_per_batch, 2), dtype=int)
        positives_hw = np.zeros_like(anchors_hw)
        # negatives_hw = np.zeros_like(anchors_hw)

        h_range = (self.patch_size // 2, H - self.patch_size // 2)
        w_range = (self.patch_size // 2, W - self.patch_size // 2)

        # First sample `samples_per_batch` anchors. Then find pos/neg patches against each of them.
        random.seed(self.random_seed)
        for batch_idx in range(B):
            for patch_idx in range(self.samples_per_batch):
                anchors_hw[batch_idx, patch_idx, :] = [
                    random.randrange(start=h_range[0], stop=h_range[1]),
                    random.randrange(start=w_range[0], stop=w_range[1]),
                ]

                # Sample nearby the anchor and check SSIM. Repeat if not similar enough.
                pos_xy_candidate = sample_hw_nearby(
                    anchors_hw[batch_idx, patch_idx, :], H=H, W=W, patch_size=self.patch_size)
                while not similar_enough(image[batch_idx, ...].cpu().detach().numpy(),
                                         h1w1=anchors_hw[batch_idx,
                                                         patch_idx, :],
                                         h2w2=pos_xy_candidate, patch_size=self.patch_size):
                    pos_xy_candidate = sample_hw_nearby(
                        anchors_hw[batch_idx, patch_idx, :], H=H, W=W, patch_size=self.patch_size)
                positives_hw[batch_idx, patch_idx, :] = pos_xy_candidate

                # # Randomly sample a patch, and it will be a negative patch.
                # negatives_hw[batch_idx, patch_idx, :] = [
                #     random.randrange(start=h_range[0], stop=h_range[1]),
                #     random.randrange(start=w_range[0], stop=w_range[1]),
                # ]

        # and anchors_hw.shape == negatives_hw.shape
        assert anchors_hw.shape == positives_hw.shape
        return anchors_hw, positives_hw  # , negatives_hw


def sample_hw_nearby(hw: Tuple[int, int], H: int, W: int, neighborhood: int = 5, patch_size: int = 7) -> Tuple[int, int]:
    return (random.randrange(start=max(hw[0]-neighborhood, patch_size),
                             stop=min(hw[0]+neighborhood, H-patch_size)),
            random.randrange(start=max(hw[1]-neighborhood, patch_size),
                             stop=min(hw[1]+neighborhood, W-patch_size)))


def similar_enough(image: np.array,
                   h1w1: Tuple[int, int], h2w2: Tuple[int, int],
                   patch_size: int, ssim_thr: float = 0.8) -> bool:
    """
    `image` dimension: [C, H, W]
    """
    patch1 = image[:,
                   h1w1[0]-patch_size//2: h1w1[0]-patch_size//2+patch_size,
                   h1w1[1]-patch_size//2: h1w1[1]-patch_size//2+patch_size]
    patch2 = image[:,
                   h2w2[0]-patch_size//2: h2w2[0]-patch_size//2+patch_size,
                   h2w2[1]-patch_size//2: h2w2[1]-patch_size//2+patch_size]

    # Channel first to channel last to accommodate SSIM.
    patch1 = np.moveaxis(patch1, 0, -1)
    patch2 = np.moveaxis(patch2, 0, -1)
    return ssim(patch1, patch2, channel_axis=-1) >= ssim_thr
