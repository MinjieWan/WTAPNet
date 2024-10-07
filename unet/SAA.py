import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from omegaconf import DictConfig


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))

        return out * identity


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.Avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Avgpool_w = nn.AdaptiveAvgPool2d((1, None))

        self.Maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.Maxpool_w = nn.AdaptiveMaxPool2d((1, None))


        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        Avgx_h = self.Avgpool_h(x)
        Avgx_w = self.Avgpool_w(x).permute(0, 1, 3, 2)
        Maxx_h = self.Maxpool_h(x)
        Maxx_w = self.Maxpool_w(x).permute(0, 1, 3, 2)

        Avgy = torch.cat([Avgx_h, Avgx_w], dim=2)
        Avgy = self.conv1(Avgy)
        Avgy = self.bn1(Avgy)
        Avgy = self.act(Avgy)

        Maxy = torch.cat([Maxx_h, Maxx_w], dim=2)
        Maxy = self.conv1(Maxy)
        Maxy = self.bn1(Maxy)
        Maxy = self.act(Maxy)

        y = Avgy + Maxy

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class SAA(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SAA, self).__init__()
        self.inp = cfg.model.n_channels
        self.hidden_dim =  72
        self.oup = 64

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(self.inp, self.hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU6(inplace=True),
            # coordinate attention
            CoordAtt(self.hidden_dim, self.hidden_dim),
            SpatialAtt(),
            # pw-linear
            nn.Conv2d(self.hidden_dim, self.oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.oup),
        )



    def forward(self, x):
        y = self.conv(x)
        return  y


