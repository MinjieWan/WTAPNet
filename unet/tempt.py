""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .resizer import *
from omegaconf import DictConfig
import hydra

class UNet_F(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UNet_F, self).__init__()
        self.n_channels = cfg.model.n_channels
        self.n_classes = cfg.model.n_classes
        self.in_resizer = Resizer(cfg)
        self.layer = cfg.model.layer
        self.mode = cfg.model.mode

        if self.mode == 1:
            if self.layer == 2:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt1(64, 128))
                self.down2 = (last_dwt1(128, 256, cfg.model.se))

                self.up2 = (part_iwt1(256, 128))
                self.up1 = (part_iwt1(128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

            if self.layer == 3:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt1(64, 128))
                self.down2 = (part_dwt1(128, 256))
                self.down3 = (last_dwt1(256, 512, cfg.model.se))

                self.up3 = (part_iwt1(512, 256))
                self.up2 = (part_iwt1(256, 128))
                self.up1 = (part_iwt1(128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

            if self.layer == 4:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt1(64, 128))
                self.down2 = (part_dwt1(128, 256))
                self.down3 = (part_dwt1(256, 512))
                self.down4 = (last_dwt1(512, 1024, cfg.model.se))
                self.up4 = (part_iwt1(1024, 512))
                self.up3 = (part_iwt1(512, 256))
                self.up2 = (part_iwt1(256, 128))
                self.up1 = (part_iwt1(128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

        if self.mode == 2:
            if self.layer == 2:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt2(256, 256))
                self.down2 = (part_dwt2(1024, 1024))

                self.up2 = (part_iwt2(512, 256))
                self.up1 = (part_iwt2(128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

            if self.layer == 3:
                self.inc = DoubleConv(cfg.model.n_channels, 32)
                self.down1 = (part_dwt2(128, 128))
                self.down2 = (part_dwt2(512, 512))
                self.down3 = (part_dwt2(2048, 2048))

                self.up3 = (part_iwt2(1024,512))
                self.up2 = (part_iwt2(256, 128))
                self.up1 = (part_iwt2(64, 32))
                self.out = (OutConv(32, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

        if self.mode == 3:
            if self.layer == 3:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt3(256, 128))
                self.down2 = (part_dwt3(512, 256))
                self.down3 = (part_dwt3(1024, 512))

                self.up3 = (part_iwt3(128, 256, 512, 256))
                self.up2 = (part_iwt3(64, 128, 256, 128))
                self.up1 = (part_iwt3(32, 64, 128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))

            if self.layer == 4:
                self.inc = DoubleConv(cfg.model.n_channels, 64)
                self.down1 = (part_dwt3(256, 128))
                self.down2 = (part_dwt3(512, 256))
                self.down3 = (part_dwt3(1024, 512))
                self.down4 = (part_dwt3(2048, 1024))
                self.up4 = (part_iwt3(256, 512, 1024, 512))
                self.up3 = (part_iwt3(128, 256, 512, 256))
                self.up2 = (part_iwt3(64, 128, 256, 128))
                self.up1 = (part_iwt3(32, 64, 128, 64))
                self.out = (OutConv(64, self.n_classes, cfg.data.size, cfg.data.interpolate_mode))


    def forward(self, x):
        x = self.in_resizer(x)
        x = self.inc(x)

        if self.mode == 1:
            if self.layer == 2:
                x1, x1_dwt = self.down1(x)
                x2, x2_dwt = self.down2(x1)
                x = self.up2(x2, x2_dwt)
                x = self.up1(x, x1_dwt)

            if self.layer == 3:
                x1, x1_dwt = self.down1(x)
                x2, x2_dwt = self.down2(x1)
                x3, x3_dwt = self.down3(x2)
                x = self.up3(x3, x3_dwt)
                x = self.up2(x, x2_dwt)
                x = self.up1(x, x1_dwt)

            if self.layer == 4:
                x1, x1_dwt = self.down1(x)
                x2, x2_dwt = self.down2(x1)
                x3, x3_dwt = self.down3(x2)
                x4, x4_dwt = self.down4(x3)
                x = self.up4(x4, x4_dwt)
                x = self.up3(x, x3_dwt)
                x = self.up2(x, x2_dwt)
                x = self.up1(x, x1_dwt)

        if self.mode == 2 or self.mode == 3:
            if self.layer == 2:
                x1 = self.down1(x)
                x2 = self.down2(x1)
                x_up = self.up2(x2, x1)
                x_up = self.up1(x_up, x)

            if self.layer == 3:
                x1 = self.down1(x)
                x2 = self.down2(x1)
                x3 = self.down3(x2)
                x_up = self.up3(x3, x2)
                x_up = self.up2(x_up, x1)
                x_up = self.up1(x_up, x)

            if self.layer == 4:
                x1 = self.down1(x)
                x2 = self.down2(x1)
                x3 = self.down3(x2)
                x4 = self.down3(x3)
                x_up = self.up3(x4, x3)
                x_up = self.up3(x_up, x2)
                x_up = self.up2(x_up, x1)
                x_up = self.up1(x_up, x)

        out = self.out(x)

        return out
