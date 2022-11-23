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
    then sample 1 positive patch for each anchor patch.

    We don't need to sample negative patches, as they can be directly inferred.
    """

    def __init__(self,
                 random_seed: int = None,
                 patch_size: int = None,
                 sampled_patches_per_image: int = None) -> None:
        self.random_seed = random_seed
        self.patch_size = patch_size
        self.sampled_patches_per_image = sampled_patches_per_image
        # Give up finding positive sample after this many unsuccessful attempts.
        self.max_attempts = 10
        self.ssim_thr = 0.5

    def sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        `image` dimension: [B, C, H, W]
        Dimension of returned Tensors: [B, samples_per_batch, 2]
        The last dimension is 2, in the (h, w) format.
        (h, w) denotes the pixel location -- not using (x, y) to avoid ambiguity against (input, GT).
        """

        B, _, H, W = image.shape

        anchors_hw = np.zeros((B, self.sampled_patches_per_image, 2),
                              dtype=int)
        positives_hw = np.zeros_like(anchors_hw)

        h_range = (self.patch_size // 2, H - self.patch_size // 2)
        w_range = (self.patch_size // 2, W - self.patch_size // 2)

        # First sample `samples_per_batch` anchors. Then find pos/neg patches against each of them.
        random.seed(self.random_seed)
        for batch_idx in range(B):
            for sample_idx in range(self.sampled_patches_per_image):
                anchors_hw[batch_idx, sample_idx, :] = [
                    random.randrange(start=h_range[0], stop=h_range[1]),
                    random.randrange(start=w_range[0], stop=w_range[1]),
                ]

                # Sample nearby the anchor and check SSIM.
                # Only samples with high enough SSIM are considered positive.
                # Keep the one with highest SSIM.
                highest_ssim = 0
                best_pos_hw_candidate = None
                for _ in range(self.max_attempts):
                    pos_hw_candidate = sample_hw_nearby(
                        anchors_hw[batch_idx, sample_idx, :],
                        H=H,
                        W=W,
                        patch_size=self.patch_size)

                    curr_ssim = compute_ssim(image[batch_idx,
                                                   ...].cpu().detach().numpy(),
                                             h1w1=anchors_hw[batch_idx,
                                                             sample_idx, :],
                                             h2w2=pos_hw_candidate,
                                             patch_size=self.patch_size)

                    if curr_ssim > self.ssim_thr and curr_ssim > highest_ssim:
                        highest_ssim = curr_ssim
                        best_pos_hw_candidate = pos_hw_candidate

                if best_pos_hw_candidate is None:
                    # If positive sample not found, just take a nearest neighbor.
                    neighbor_hw = anchors_hw[batch_idx, sample_idx, :]
                    neighbor_hw[0] = neighbor_hw[0] - 1 if neighbor_hw[
                        0] > H // 2 else neighbor_hw[0] + 1
                    neighbor_hw[1] = neighbor_hw[1] - 1 if neighbor_hw[
                        1] > W // 2 else neighbor_hw[1] + 1
                    best_pos_hw_candidate = neighbor_hw

                positives_hw[batch_idx, sample_idx, :] = best_pos_hw_candidate

        assert anchors_hw.shape == positives_hw.shape
        return anchors_hw, positives_hw


def sample_hw_nearby(hw: Tuple[int, int],
                     H: int,
                     W: int,
                     neighborhood: int = 5,
                     patch_size: int = 7) -> Tuple[int, int]:
    h_start = max(hw[0] - neighborhood, patch_size // 2)
    h_stop = min(hw[0] + neighborhood, H - patch_size // 2)
    w_start = max(hw[1] - neighborhood, patch_size // 2)
    w_stop = min(hw[1] + neighborhood, W - patch_size // 2)

    return (random.randrange(start=h_start, stop=h_stop),
            random.randrange(start=w_start, stop=w_stop))


def compute_ssim(image: np.array, h1w1: Tuple[int, int], h2w2: Tuple[int, int],
                 patch_size: int) -> float:
    """
    `image` dimension: [C, H, W]
    """
    patch1 = image[:, h1w1[0] - patch_size // 2:h1w1[0] - patch_size // 2 +
                   patch_size, h1w1[1] - patch_size // 2:h1w1[1] -
                   patch_size // 2 + patch_size]
    patch2 = image[:, h2w2[0] - patch_size // 2:h2w2[0] - patch_size // 2 +
                   patch_size, h2w2[1] - patch_size // 2:h2w2[1] -
                   patch_size // 2 + patch_size]

    # Channel first to channel last to accommodate SSIM.
    patch1 = np.moveaxis(patch1, 0, -1)
    patch2 = np.moveaxis(patch2, 0, -1)
    return ssim(patch1, patch2, channel_axis=-1)
