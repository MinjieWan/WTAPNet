""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .resizer import SEModule

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return  x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class DWT_d4(nn.Module):
    def __init__(self):
        super(DWT_d4, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init_d4(x)

def dwt_init_d4(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class last_dwt1(nn.Module):
    def __init__(self, in_channels, out_channels, se):
        super().__init__()
        self.dwt = DWT_d4()
        # if se:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         SEModule(out_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_dwt = self.dwt(x)
       # out = self.conv(x_ll)

        return x_dwt

class part_dwt1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_ll)

        return out, x_dwt

class part_dwt2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_dwt)

        return out

class part_dwt3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_dwt)

        return out


class part_iwt1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels, out_channels)
        self.iwt = IWT()
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)
        x2 = self.iwt(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x1, x2)
        return self.conv(x)

class part_iwt1_fuse(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(size=48, mode='bilinear', align_corners=True)
            self.conv_mid = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels , out_channels)
        self.iwt = IWT()
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)
        x2 = self.iwt(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x1, x2)
        return self.conv(x)

class part_iwt2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.iwt = IWT()

    def forward(self, x1, x2):
        x1 = self.iwt(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class part_iwt3(nn.Module):
    def __init__(self, mid_in, mid_out, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up =IWT()
        self.conv_mid = nn.Conv2d(mid_in, mid_out, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_mid(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, size, interpolate_mode):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.interpolate = partial(F.interpolate,
                                   size=size,
                                   mode=interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.interpolate(x)
        return x

class out_conv(nn.Module):
    def __init__(self, ch1, ch2, n_classes):
        super().__init__()
      #  self.mid_conv = nn.Conv2d(ch2, ch1, kernel_size=1)
        self.out_conv = nn.Conv2d(ch1+ch2, n_classes, 1)
      #  self.fuse = fuse(ch1)
        
    def forward(self, x1, x2):
            x2 = F.upsample(x2, size=x1.shape[2:], mode='bilinear')
            #x2 = self.mid_conv(x2)
            
           # out = self.fuse(x1, x2)
            # out = torch.add(x1, x2)
            out = torch.cat((x1,x2), dim=1)
            out = self.out_conv(out)
            return out
    
class fuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(fuse,self).__init__()
        self.channels = channels
        self.mid_channels = int(channels // r)

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )


    def forward(self, xh, xl):
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * (xl * topdown_wei) + 2 * (xh * bottomup_wei)
        xs = self.post(xs)

        return xs
    
    
    
