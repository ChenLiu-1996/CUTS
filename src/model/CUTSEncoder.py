import torch.nn as nn

from ..module import ConvBlock


class CUTSEncoder(nn.Module):
    def __init__(self):
        super(CUTSEncoder, self).__init__()
        self.conv1 = ConvBlock(3, 32)

    def forward(self, x):
        x1_, patchify_loss = self.conv1(x)

        return x1_, patchify_loss
