import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PatchSampler


class CUTSEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, num_kernels: int = 32, random_seed: int = None):
        super(CUTSEncoder, self).__init__()

        self.channels = in_channels

        self.batch_norm1 = nn.BatchNorm2d(num_kernels)
        self.batch_norm2 = nn.BatchNorm2d(num_kernels*2)
        self.batch_norm3 = nn.BatchNorm2d(num_kernels*4)

        self.conv1 = nn.Conv2d(in_channels, num_kernels,
                               kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(num_kernels, num_kernels*2,
                               kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(num_kernels*2, num_kernels*4,
                               kernel_size=3, padding='same')

        self.patch_sampler = PatchSampler(random_seed=random_seed)

        self.patch_size = 7  # Encoder includes 7 x 7 neighbor information

        # Reconstruction module.
        self.recon = CUTSRecon(channels=self.channels,
                               patch_size=self.patch_size)

    def forward(self, image):
        # `x` is the image input.
        B, C, _, _ = image.shape

        z = F.leaky_relu(self.batch_norm1(self.conv1(image)))
        z = F.leaky_relu(self.batch_norm2(self.conv2(z)))
        z = F.leaky_relu(self.batch_norm3(self.conv3(z)))

        # At this point, `z` is the latent projection of x.
        # We deliberately chose the layers such that the image-space resolution is the same.
        # In this way, we can (roughly) claim that z[i][j] is a feature representation
        # of the image patch centered at x[i][j].

        # Sample patches for contrastive loss & reconstruction loss
        anchors_xy, positives_xy, negatives_xy = self.patch_sampler.sample(
            image)

        anchor_patches = torch.zeros(
            (B, C, self.patch_size, self.patch_size), requires_grad=False).to(image.device)
        positive_patches = torch.zeros_like(anchor_patches).to(image.device)
        negative_patches = torch.zeros_like(anchor_patches).to(image.device)
        encoded_anchor_patches = torch.zeros(
            (B, z.shape[1]), requires_grad=False).to(image.device)

        assert anchors_xy.shape[0] == B
        for batch_idx in range(B):
            for patch_idx in range(anchors_xy.shape[1]):
                anchor_patches[batch_idx, ...] = image[batch_idx, :,
                                                       anchors_xy[batch_idx, patch_idx, 0]-self.patch_size//2:
                                                       anchors_xy[batch_idx, patch_idx, 0] -
                                                           self.patch_size//2+self.patch_size,
                                                       anchors_xy[batch_idx, patch_idx, 1]-self.patch_size//2:
                                                       anchors_xy[batch_idx, patch_idx, 1]-self.patch_size//2+self.patch_size]
                positive_patches[batch_idx, ...] = image[batch_idx, :,
                                                         positives_xy[batch_idx, patch_idx, 0]-self.patch_size//2:
                                                         positives_xy[batch_idx, patch_idx, 0] -
                                                             self.patch_size//2+self.patch_size,
                                                         positives_xy[batch_idx, patch_idx, 1]-self.patch_size//2:
                                                         positives_xy[batch_idx, patch_idx, 1]-self.patch_size//2+self.patch_size]
                negative_patches[batch_idx, ...] = image[batch_idx, :,
                                                         negatives_xy[batch_idx, patch_idx, 0]-self.patch_size//2:
                                                         negatives_xy[batch_idx, patch_idx, 0] -
                                                             self.patch_size//2+self.patch_size,
                                                         negatives_xy[batch_idx, patch_idx, 1]-self.patch_size//2:
                                                         negatives_xy[batch_idx, patch_idx, 1]-self.patch_size//2+self.patch_size]
                encoded_anchor_patches[batch_idx, ...] = z[batch_idx, :,
                                                           negatives_xy[batch_idx,
                                                                        patch_idx, 0],
                                                           negatives_xy[batch_idx,
                                                                        patch_idx, 1]
                                                           ]

        # Reconstruction.
        # Reshape to [B, -1]
        x = torch.clone(anchor_patches)  # `anchor_patches` is used later.
        x = x.reshape(x.shape[0], -1)
        z = encoded_anchor_patches
        z = z.reshape(z.shape[0], -1)

        x_recon = self.recon(z)
        print(x.shape, x_recon.shape)

        # patchify_target = []
        # for i in range(x.shape[0]):
        #     tmp = x[i]

        #     tmp = F.pad(tmp, pad=[self.k // 2, self.k //
        #                 2, self.k // 2, self.k // 2], value=0)

        #     tmp = tmp.unfold(1, self.k, 1).unfold(2, self.k, 1).reshape(
        #         self.channels, -1, self.k, self.k).permute(1, 0, 2, 3)
        #     tmp = tmp.reshape([tmp.shape[0], -1])
        #     patchify_target.append(tmp)
        # patchify_target = torch.stack(patchify_target, dim=0)
        # patchify_loss = (
        #     (torch.tanh(self.recon(x3)) - patchify_target)**2).mean()

        return anchor_patches, positive_patches, negative_patches, encoded_anchor_patches


class CUTSRecon(nn.Module):
    """
    The low-budget reconstruction network, which is simply a linear layer.
    # TODO: Consider a convolutional reconstruction module.
    """

    def __init__(self, channels: int = None, patch_size: int = None):
        super(CUTSRecon, self).__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.recon = nn.Linear(128, self.channels * self.patch_size**2)

    def forward(self, z):
        return self.recon(z)
