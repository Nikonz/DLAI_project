import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import GConvZ2Block, GConvGBlock

class GAllCNNC(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()

        self.mode = 'classifier'

        self.layers1 = nn.Sequential(
            nn.Dropout(0.2),
            GConvZ2Block(in_channels=in_channels, out_channels=17, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=17, out_channels=17, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=17, out_channels=17, kernel_size=3, stride=2, padding=1),
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(0.5),
            GConvGBlock(in_channels=17, out_channels=34, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=34, out_channels=34, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=34, out_channels=34, kernel_size=3, stride=2, padding=1),
        )
        self.layers3 = nn.Sequential(
            nn.Dropout(0.5),
            GConvGBlock(in_channels=34, out_channels=34, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=34, out_channels=34, kernel_size=1, stride=1, padding=0),
            GConvGBlock(in_channels=34, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)

        x = torch.mean(x, axis=-1)
        x = torch.mean(x, axis=-1)
        x = torch.mean(x, axis=-1)

        return x

class GAllCNNCRotPool(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()

        self.mode = 'classifier'

        self.layers1 = nn.Sequential(
            nn.Dropout(0.2),
            GConvZ2Block(in_channels=in_channels, out_channels=20, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=1),
        )
        self.layers2 = nn.Sequential(
            nn.Dropout(0.5),
            GConvZ2Block(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=40, out_channels=40, kernel_size=3, stride=2, padding=1),
        )
        self.layers3 = nn.Sequential(
            nn.Dropout(0.5),
            GConvZ2Block(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1),
            GConvGBlock(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0),
            GConvGBlock(in_channels=40, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.layers1(x)
        x = torch.max(x, axis=-3)[0]
        x = self.layers2(x)
        x = torch.max(x, axis=-3)[0]
        x = self.layers3(x)
        x = torch.max(x, axis=-3)[0]

        x = torch.mean(x, axis=-1)
        x = torch.mean(x, axis=-1)

        return x
