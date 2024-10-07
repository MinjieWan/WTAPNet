import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from omegaconf import DictConfig
import hydra


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class RSF(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(RSF, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




class Resizer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        input_channel = cfg.model.n_channels
        head_channel = 32

        # 0.88
        # head_channel = 16

        # first
        modules_head = [
            nn.Conv2d(input_channel, head_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_channel),
            nn.ReLU(inplace=True)]

        self.head = nn.Sequential(*modules_head)

        # building RSF blocks
        # modules_body = nn.ModuleList()
        # # input channel, output channel, kernal, stride, expend channel, se, activate
        # modules_body.append(RSF(16, 16, 3, 1, 16, True, nl='RE'))
        # modules_body.append(RSF(16, 24, 3, 1, 72, True, nl='RE'))
        # modules_body.append(RSF(24, 24, 3, 1, 88, True, nl='RE'))
        # modules_body.append(RSF(24, 32, 5, 1, 96, True, nl='RE'))
        # modules_body.append(RSF(32, 32, 3, 1, 96, True, nl='RE'))
        # modules_body.append(RSF(32, 64, 3, 1, 96, True, nl='RE'))
        # modules_body.append(RSF(64, 64, 3, 1, 128, True, nl='RE'))
        # modules_body.append(RSF(64, 64, 3, 1, 128, True, nl='RE'))
        # self.body = nn.Sequential(*modules_body)


        # input channel, output channel, kernal, stride, expend channel, se, activate
        # self.modules_body1 =(RSF(16, 16, 3, 1, 16, True, nl='RE'))
        # self.modules_body2 = (RSF(16, 24, 3, 1, 72, True, nl='RE'))
        # self.modules_body3 = (RSF(40, 40, 3, 1, 96, True, nl='RE'))
        # self.modules_body4 = (RSF(40, 40, 3, 1, 120, True, nl='RE'))
        # self.modules_body5 = (RSF(40, 48, 3, 1, 144, True, nl='RE'))
        # self.modules_body6 = (RSF(88, 88, 3, 1, 240, True, nl='RE'))
        # self.modules_body7 = (RSF(88, 88, 3, 1, 288, True, nl='RE'))
        # self.modules_body8 = (RSF(88, 88, 3, 1, 480, True, nl='RE'))

        # 0.88
        # self.modules_body1 = (RSF(16, 16, 3, 1, 40, True, nl='RE'))
        # self.modules_body2 = (RSF(16, 32, 3, 1, 72, True, nl='RE'))
        # self.modules_body3 = (RSF(32, 64, 3, 1, 96, True, nl='RE'))
        # self.modules_body4 = (RSF(64, 64, 3, 1, 128, True, nl='RE'))

        # 0.88(稳定)
        self.modules_body1 = (RSF(32, 64, 3, 1, 64, True, nl='RE'))
        self.modules_body2 = (RSF(64, 64, 3, 1, 72, True, nl='RE'))
        self.modules_body3 = (RSF(64, 88, 3, 1, 96, True, nl='RE'))
        self.modules_body4 = (RSF(88, 88, 3, 1, 128, True, nl='RE'))

        # self.modules_body1 = (ResBlock(32))
        # self.modules_body2 = (ResBlock(32))
        # self.modules_body3 = (ResBlock(32))
        # self.modules_body4 = (ResBlock(32))





        #0.88→(64,1)
        # define tail module
        modules_tail = []
        modules_tail.append(nn.Conv2d(88, cfg.resizer.scale**2*88, 1, padding=0, stride=1))
        # modules_tail.append(nn.Conv2d(96, 288, 1, padding=0, stride=1))
        modules_tail.append(nn.PixelShuffle(cfg.resizer.scale))

        if cfg.resizer.last_bn is True:
            modules_tail.append(nn.BatchNorm2d(88))
        if cfg.resizer.last_act == 'relu':
            modules_tail.append(nn.ReLU(True))

        modules_tail.append(nn.Conv2d(88, input_channel, 1, padding=0, stride=1))

        self.tail = nn.Sequential(*modules_tail)
        # self.resconv = nn.Conv2d(input_channel, 64, 1, padding=0, stride=1)
        self.interpolate = partial(F.interpolate,
                                   size=cfg.data.scale_input * cfg.resizer.scale,
                                   mode=cfg.resizer.mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        identity = x

        x = self.head(x)

        # out1 = self.modules_body1(x)
        # out2 = self.modules_body2(out1)
        # out3 = self.modules_body3(torch.cat([out1, out2], dim=1))
        # out4 = self.modules_body4(out3)
        # out5 = self.modules_body5(out4)
        # out6 = self.modules_body6(torch.cat([out4, out5], dim=1))
        # out7 = self.modules_body7(out6)
        # out = self.modules_body8(out7)
        #
        # out1 = self.modules_body1(x)
        # out2 = self.modules_body2(out1)
        # out3 = self.modules_body3(out2)
        # out = self.modules_body4(out3)

        out1 = self.modules_body1(x)
        out2 = self.modules_body2(out1)
        out3 = self.modules_body3(out2)


        out = self.modules_body4(out3)


        out = self.tail(out)

        identity = self.interpolate(identity)
        #identity = self.resconv(identity)

        return out + identity