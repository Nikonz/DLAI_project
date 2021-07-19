import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import GConvZ2Block, GConvGBlock

class GUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, levels=3,
            first_hidden_channels=64):
        super().__init__()

        self.mode = 'segmentator'

        conv = []
        deconv = []

        in_ch = in_channels
        out_ch = first_hidden_channels

        for idx in range(levels):
            conv += [
                    *GConvZ2Block(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1).layers(),
                    *GConvGBlock(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1).layers(),
                    ]
            if idx < levels - 1:
                conv.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch
            out_ch *= 2

        bottleneck = GConvGBlock(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1)

        in_ch *= 2
        out_ch = in_ch // 4

        for _ in range(levels - 1):
            deconv += [
                      nn.Upsample(scale_factor=2, mode='bilinear'),
                      *GConvZ2Block(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=1, padding=1).layers(),
                      *GConvGBlock(in_channels=in_ch // 2, out_channels=out_ch, kernel_size=3, stride=1, padding=1).layers(),
                      ]
            in_ch //= 2
            out_ch //= 2

        self.conv = nn.Sequential(*conv)
        self.bottleneck = bottleneck
        self.deconv = nn.Sequential(*deconv)
        self.out = GConvGBlock(in_channels=first_hidden_channels,
                out_channels=out_channels, kernel_size=1, batch_norm=None,
                activation=False)

    def forward(self, x):
        r = []
        for idx, c in enumerate(self.conv):
            x = c(x)
            if idx % 7 == 5:
                x_pooled = torch.max(x, axis=-3)[0]
                r.append(x_pooled)
                if idx < len(self.conv) - 1:
                    x = x_pooled
        x = self.bottleneck(x)

        r = r[::-1]
        for idx, d in enumerate(self.deconv):
            if idx % 7 == 0:
                x = torch.max(x, axis=-3)[0]
                x = torch.cat((x, r[idx // 7]), dim=1)
            x = d(x)
        x = self.out(x)
        x = torch.max(x, axis=-3)[0]
        return x

class GUnetGSkipConn(GUnet):
    def __init__(self, in_channels=3, out_channels=1, levels=3,
            first_hidden_channels=64):
        super().__init__(in_channels, out_channels, levels,
                first_hidden_channels)

    def forward(self, x):
        r = []
        for idx, c in enumerate(self.conv):
            x = c(x)
            if idx % 7 == 5:
                r.append(x)
                x_pooled = torch.max(x, axis=-3)[0]
                if idx < len(self.conv) - 1:
                    x = x_pooled
        x = self.bottleneck(x)

        r = r[::-1]
        for idx, d in enumerate(self.deconv):
            if idx % 7 == 0:
                x = torch.cat((x, r[idx // 7]), dim=1)
                x = torch.max(x, axis=-3)[0]
            x = d(x)
        x = self.out(x)
        x = torch.max(x, axis=-3)[0]
        return x
