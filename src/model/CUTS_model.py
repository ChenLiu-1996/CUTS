import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import PatchSampler


class CUTSEncoder(nn.Module):
    """
    The CUTS model.

    `inference`: if set to True, will only produce the latent embedding `z` and skip other stuff.
    """

    def __init__(self, in_channels: int = 3, num_kernels: int = None, random_seed: int = None,
                 sampled_patches_per_image: int = None, inference: bool = False) -> None:
        super(CUTSEncoder, self).__init__()

        self.channels = in_channels

        self.batch_norm1 = nn.BatchNorm2d(num_kernels)
        self.batch_norm2 = nn.BatchNorm2d(num_kernels*2)
        self.batch_norm3 = nn.BatchNorm2d(num_kernels*4)
        self.batch_norm4 = nn.BatchNorm2d(num_kernels*8)

        self.conv1 = nn.Conv2d(in_channels, num_kernels,
                               kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(num_kernels, num_kernels*2,
                               kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(num_kernels*2, num_kernels*4,
                               kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(num_kernels*4, num_kernels*8,
                               kernel_size=3, padding='same')
        self.latent_dim = num_kernels*8

        # Encoder includes 9 x 9 neighbor information
        # (3x3 kernel, 4 layers, receptive field is 9x9).
        self.patch_size = 9

        self.inference = inference

        self.patch_sampler = PatchSampler(
            random_seed=random_seed, patch_size=self.patch_size, sampled_patches_per_image=sampled_patches_per_image)

        # Reconstruction module.
        self.recon = CUTSRecon(channels=self.channels,
                               patch_size=self.patch_size,
                               latent_dim=self.latent_dim,
                               sampled_patches_per_image=sampled_patches_per_image)

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        return

    def load_weights(self, model_save_path: str) -> None:
        self.load_state_dict(torch.load(model_save_path))
        return

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Dimension acronyms:
            B: batch size
            S: number of sampled patches per image
            C: image channel (i.e., 3 for R/G/B image)
            H: image height
            W: image width
            L: latent vector dimension
            P: patch size (h == w)
        """
        # `x` is the image input.
        B, C, _, _ = x.shape

        z = F.leaky_relu(self.batch_norm1(self.conv1(x)))
        z = F.leaky_relu(self.batch_norm2(self.conv2(z)))
        z = F.leaky_relu(self.batch_norm3(self.conv3(z)))
        z = F.leaky_relu(self.batch_norm4(self.conv4(z)))

        # At this point, `z` is the latent projection of x.
        # We deliberately chose the layers such that the image-space resolution is the same.
        # In this way, we can (roughly) claim that z[i][j] is a feature representation
        # of the image patch centered at x[i][j].

        if not self.inference:
            '''
            Sample patches for contrastive loss & reconstruction loss.
            Patches are `x_anchors`.
            Latent vectors of patches are `z_anchors`.
            Latent vectors of positive pairs are `z_positives`.
            Negative pairs are implied from non-matching indices of `z_anchors` and `z_positives`.
                `x`:           [B, C, H, W]
                `z`:           [B, L, H, W]
                `x_anchors`:   [B, S, C, P, P]
                `z_anchors`:   [B, S, L]
                `z_positives`: [B, S, L]
            '''
            anchors_hw, positives_hw = self.patch_sampler.sample(x)
            # num samples per batch
            S = anchors_hw.shape[1]

            # These are containers that will later hold the torch.Tensors.
            # They won't track gradients themselves but the Tensors they contain will.
            x_anchors = torch.zeros(
                (B, S, C, self.patch_size, self.patch_size)).to(x.device)
            z_anchors = torch.zeros((B, S, self.latent_dim)).to(x.device)
            z_positives = torch.zeros_like(z_anchors).to(x.device)

            assert anchors_hw.shape[0] == B
            for batch_idx in range(B):
                for sample_idx in range(S):
                    x_anchors[batch_idx, sample_idx, ...] = x[batch_idx, :,
                                                              anchors_hw[batch_idx, sample_idx, 0]-self.patch_size//2:
                                                              anchors_hw[batch_idx, sample_idx, 0] -
                                                              self.patch_size//2+self.patch_size,
                                                              anchors_hw[batch_idx, sample_idx, 1]-self.patch_size//2:
                                                              anchors_hw[batch_idx, sample_idx, 1] -
                                                              self.patch_size//2+self.patch_size
                                                              ]
                    z_anchors[batch_idx, sample_idx, ...] = z[batch_idx, :,
                                                              anchors_hw[batch_idx,
                                                                         sample_idx, 0],
                                                              anchors_hw[batch_idx,
                                                                         sample_idx, 1]
                                                              ]
                    z_positives[batch_idx, sample_idx, ...] = z[batch_idx, :,
                                                                positives_hw[batch_idx,
                                                                             sample_idx, 0],
                                                                positives_hw[batch_idx,
                                                                             sample_idx, 1]
                                                                ]

            # Reconstruction.
            # Reshape to [B, -1]
            x_anchors = x_anchors.reshape(B, -1)
            x_recon = self.recon(z_anchors.reshape(B, -1))

            return z, x_anchors, x_recon, z_anchors, z_positives
        else:
            return z


class CUTSRecon(nn.Module):
    """
    The low-budget reconstruction network, which is simply a linear layer.
    # TODO: Consider a convolutional reconstruction module.
    """

    def __init__(self, channels: int = None, patch_size: int = None,
                 latent_dim: int = None, sampled_patches_per_image: int = None) -> None:
        super(CUTSRecon, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        # Maps [B, S, L] to [B, S, C, H, W].
        self.recon = nn.Linear(
            sampled_patches_per_image * latent_dim,
            sampled_patches_per_image * self.channels * self.patch_size**2)

    def forward(self, z):
        return self.recon(z)
