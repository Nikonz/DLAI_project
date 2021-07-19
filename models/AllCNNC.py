import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBlock

class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)

class AllCNNC(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()

        self.mode = 'classifier'

        self.layers1 = nn.Sequential(
            nn.Dropout(0.2),
            ConvBlock(in_channels=in_channels, out_channels=48, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, 2, 1),
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(0.5),
            ConvBlock(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(3, 2, 1),
        )
        self.layers3 = nn.Sequential(
            nn.Dropout(0.5),
            ConvBlock(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=96, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)

        x = torch.mean(x, axis=-1)
        x = torch.mean(x, axis=-1)
        return x

class AllCNNCChanPool(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()

        self.mode = 'classifier'

        self.layers1 = nn.Sequential(
            nn.Dropout(0.2),
            ConvBlock(in_channels=in_channels, out_channels=54, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=54, out_channels=54, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=54, out_channels=54, kernel_size=3, stride=2, padding=1),
            ChannelPool(kernel_size=3, stride=3)
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(0.5),
            ConvBlock(in_channels=18, out_channels=108, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=108, out_channels=108, kernel_size=3, stride=2, padding=1),
            ChannelPool(kernel_size=3, stride=3)
        )
        self.layers3 = nn.Sequential(
            nn.Dropout(0.5),
            ConvBlock(in_channels=36, out_channels=108, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=108, out_channels=108, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=108, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)

        x = torch.mean(x, axis=-1)
        x = torch.mean(x, axis=-1)
        return x
