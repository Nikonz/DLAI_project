import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4MConvP4M, P4MConvZ2

class BaseConvBlock(nn.Module):
    def __init__(self, out_channels, batch_norm='2d', activation=True):
        super().__init__()

        self.conv = None
        self.batch_norm = None
        self.activation = None

        if batch_norm == '2d':
            self.norm = nn.BatchNorm2d(out_channels)
        elif batch_norm == '3d':
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = None

        if activation:
            self.activ = nn.ReLU()
        else:
            self.activ = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activ is not None:
            x = self.activ(x)
        return x

    def layers(self):
        return self.conv, self.norm, self.activ

class ConvBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, batch_norm='2d', activation=True):
        super().__init__(out_channels, batch_norm=batch_norm, activation=activation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

class GConvZ2Block(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, batch_norm='3d', activation=True):
        super().__init__(out_channels, batch_norm=batch_norm, activation=activation)
        self.conv = P4MConvZ2(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

class GConvGBlock(BaseConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, batch_norm='3d', activation=True):
        super().__init__(out_channels, batch_norm=batch_norm, activation=activation)
        self.conv = P4MConvP4M(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
