import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, nfilt, k=3):
        super(ConvBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(nfilt)
        self.batch_norm2 = nn.BatchNorm2d(nfilt*4)
        self.batch_norm3 = nn.BatchNorm2d(nfilt*8)

        self.conv1 = nn.Conv2d(
            in_channels, nfilt, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(nfilt, nfilt*4, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(nfilt*4, nfilt*8, kernel_size=3, padding='same')

        self.l1 = nn.Linear(nfilt*8, 128)

        self.k = k
        self.channels = in_channels
        self.l2 = nn.Linear(128, self.k * self.k * self.channels)

    def forward(self, x):
        x1 = F.leaky_relu(self.batch_norm1(self.conv1(x)))
        x2 = F.leaky_relu(self.batch_norm2(self.conv2(x1)))
        x3 = F.leaky_relu(self.batch_norm3(self.conv3(x2)))

        x3 = x3.permute(0, 2, 3, 1)
        x3 = x3.reshape(x3.shape[0], -1, x3.shape[-1])
        x3 = self.l1(x3)

        patchify_target = []
        for i in range(x.shape[0]):
            tmp = x[i]

            tmp = F.pad(tmp, pad=[self.k // 2, self.k //
                        2, self.k // 2, self.k // 2], value=0)

            tmp = tmp.unfold(1, self.k, 1).unfold(2, self.k, 1).reshape(
                self.channels, -1, self.k, self.k).permute(1, 0, 2, 3)
            tmp = tmp.reshape([tmp.shape[0], -1])
            patchify_target.append(tmp)
        patchify_target = torch.stack(patchify_target, dim=0)
        patchify_loss = ((torch.tanh(self.l2(x3)) - patchify_target)**2).mean()

        return x3, patchify_loss
