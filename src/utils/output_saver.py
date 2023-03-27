import os
import warnings
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


class OutputSaver(object):
    """
    A numpy file saver...
    We off-sourced most of the segmentation and analysis to other files under `script_analysis`.

    @params:
    `save_path`:
        Path to save numpy files and image files.
    """

    def __init__(self, save_path: str = None, random_seed: int = None) -> None:
        self.random_seed = random_seed

        self.save_path_numpy = '%s/%s/' % (save_path, 'numpy_files')
        os.makedirs(self.save_path_numpy, exist_ok=True)
        self.image_idx = 0

    def save(
        self,
        image_batch: torch.Tensor,
        recon_batch: torch.Tensor,
        label_true_batch: torch.Tensor,
        latent_batch: torch.Tensor,
    ) -> Tuple[float, np.array, np.array]:
        """
        @params:
        `label_true`: ground truth segmentation map.
        `latent`: latent embedding by the model.

        `label_true` and `latent` are expected to have dimension [B, H, W, C].
        """

        image_batch = image_batch.cpu().detach().numpy()
        recon_batch = recon_batch.cpu().detach().numpy()
        if label_true_batch is not None:
            label_true_batch = label_true_batch.cpu().detach().numpy()
        latent_batch = latent_batch.cpu().detach().numpy()
        # channel-first to channel-last
        image_batch = np.moveaxis(image_batch, 1, -1)
        recon_batch = np.moveaxis(recon_batch, 1, -1)
        if label_true_batch is not None:
            label_true_batch = np.moveaxis(label_true_batch, 1, -1)
        latent_batch = np.moveaxis(latent_batch, 1, -1)

        # Squeeze excessive dimension.
        image_batch = squeeze_excessive_dimension(image_batch)
        recon_batch = squeeze_excessive_dimension(recon_batch)
        if label_true_batch is not None:
            label_true_batch = squeeze_excessive_dimension(label_true_batch)

        B, H, W, C = latent_batch.shape

        if label_true_batch is None:
            label_true_batch = np.empty((B,H,W))
            label_true_batch[:] = np.nan

        # Save the images, labels, and latent embeddings as numpy files for future reference.
        for image_idx in tqdm(range(B)):
            self.save_as_numpy(
                image=image_batch[image_idx, ...],
                recon=recon_batch[image_idx, ...],
                label=label_true_batch[image_idx, ...],
                # [H, W, C] to [H x W, C]
                latent=latent_batch[image_idx, ...].reshape((H * W, C)))
        return

    def save_as_numpy(self, image: np.array, recon: np.array,
                      label: np.array, latent: np.array) -> None:
        with open(
                '%s/%s' %
            (self.save_path_numpy, 'sample_%s.npz' % str(self.image_idx).zfill(5)),
                'wb+') as f:
            np.savez(f, image=image, recon=recon, label=label, latent=latent)
        self.image_idx += 1


def squeeze_excessive_dimension(batched_data: np.array) -> np.array:
    assert len(batched_data.shape) in [3, 4]
    if len(batched_data.shape) == 4 and batched_data.shape[-1] == 1:
        # (B, H, W, 1) to (B, H, W)
        batched_data = batched_data.reshape(batched_data.shape[:3])
    return batched_data