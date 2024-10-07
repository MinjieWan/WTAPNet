""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .resizer import SEModule
import pywt
import numpy as np

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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class last_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
         #   DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_fuse(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_mid = DoubleConv(in_channels, out_channels)

        self.fuse = fuse(out_channels)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_mid(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.fuse(x1, x2)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.fuse = fuse(in_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.fuse(x1, x2)
        return self.conv(x)


class DWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wavelet):
        B, C, H, W = x.shape
        coeffs_tensor = []
        coeffs_LL = []
        dtype = x.dtype
        device = x.device

        for b in range(B):
            batch_coeffs = []
            batch_LL = []
            for c in range(C):
                # Extract the channel image
                img = x[b, c].detach().cpu().numpy()

                # Apply DWT
                coeffs2 = pywt.dwt2(img, wavelet)
                cA, (cH, cV, cD) = coeffs2

                # Convert to tensors and concatenate along the channel dimension
                cA_tensor = torch.tensor(cA, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cH_tensor = torch.tensor(cH, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cV_tensor = torch.tensor(cV, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cD_tensor = torch.tensor(cD, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)

                # Concatenate all coefficients
                combined = torch.cat((cA_tensor, cH_tensor, cV_tensor, cD_tensor), dim=0)  # Shape (4, H/2, W/2)
                batch_coeffs.append(combined)
                batch_LL.append(cA_tensor)

            batch_coeffs_tensor = torch.cat(batch_coeffs, dim=0).unsqueeze(0)  # Shape (1, 4*C, H/2, W/2)
            coeffs_tensor.append(batch_coeffs_tensor)
            batch_LL_tensor = torch.cat(batch_LL, dim=0)  # Shape (4*C, H/2, W/2)
            coeffs_LL.append(batch_LL_tensor.unsqueeze(0))

        coeffs_tensor = torch.cat(coeffs_tensor, dim=0)  # Shape (B, 4*C, H/2, W/2)
        coeffs_LL = torch.cat(coeffs_LL, dim=0)

        ctx.save_for_backward(x, coeffs_tensor, coeffs_LL)
        ctx.wavelet = wavelet

        return coeffs_LL, coeffs_tensor

    @staticmethod
    def backward(ctx, grad_coeffs_LL, grad_coeffs_tensor):
        x, coeffs_tensor, coeffs_LL = ctx.saved_tensors
        wavelet = ctx.wavelet

        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2

        grad_input = torch.zeros_like(x)

        for b in range(B):
            for c in range(C):
                # Extract gradients for each subband
                grad_cA = grad_coeffs_LL[b, c].cpu().numpy()
                grad_cH = grad_coeffs_tensor[b, c * 4 + 1].cpu().numpy()
                grad_cV = grad_coeffs_tensor[b, c * 4 + 2].cpu().numpy()
                grad_cD = grad_coeffs_tensor[b, c * 4 + 3].cpu().numpy()

                # Apply inverse DWT to propagate the gradients
                grad_img = pywt.idwt2((grad_cA, (grad_cH, grad_cV, grad_cD)), wavelet)
                grad_img_tensor = torch.tensor(grad_img, device=x.device, dtype=x.dtype)

                # Ensure the gradient image matches the input size
                grad_img_tensor = grad_img_tensor[:H, :W]  # Ensure the size matches

                grad_input[b, c] = grad_img_tensor

        return grad_input, None


class IWTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeffs_tensor, wavelet):
        B, C4, H2, W2 = coeffs_tensor.shape
        C = C4 // 4
        device = coeffs_tensor.device
        dtype = coeffs_tensor.dtype
        restored_images = []

        for b in range(B):
            batch_images = []
            for c in range(C):
                # Extract the coefficients
                cA = coeffs_tensor[b, c * 4 + 0].cpu().numpy()
                cH = coeffs_tensor[b, c * 4 + 1].cpu().numpy()
                cV = coeffs_tensor[b, c * 4 + 2].cpu().numpy()
                cD = coeffs_tensor[b, c * 4 + 3].cpu().numpy()

                # Apply IWT
                restored_img = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
                batch_images.append(torch.tensor(restored_img, device=device, dtype=dtype))

            restored_images.append(torch.stack(batch_images))

        ctx.save_for_backward(coeffs_tensor)
        ctx.wavelet = wavelet

        return torch.stack(restored_images)

    @staticmethod
    def backward(ctx, grad_output):
        coeffs_tensor, = ctx.saved_tensors
        wavelet = ctx.wavelet

        B, C4, H2, W2 = coeffs_tensor.shape
        C = C4 // 4
        device = coeffs_tensor.device
        dtype = coeffs_tensor.dtype
        restored_grad = torch.zeros_like(coeffs_tensor)

        for b in range(B):
            for c in range(C):
                # Extract the coefficients from the gradient tensor
                restored_cA, (restored_cH, restored_cV, restored_cD) = pywt.dwt2(grad_output[b, c].cpu().numpy(), wavelet)

                restored_grad[b, c * 4 + 0] = torch.tensor(restored_cA, device=device, dtype=dtype)
                restored_grad[b, c * 4 + 1] = torch.tensor(restored_cH, device=device, dtype=dtype)
                restored_grad[b, c * 4 + 2] = torch.tensor(restored_cV, device=device, dtype=dtype)
                restored_grad[b, c * 4 + 3] = torch.tensor(restored_cD, device=device, dtype=dtype)

        return restored_grad, None


class DWT(nn.Module):
    def __init__(self, wavelet='coif1'):
        super(DWT, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        return DWTFunction.apply(x, self.wavelet)

class DWT_d4(nn.Module):
    def __init__(self, wavelet='coif1'):
        super(DWT_d4, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        return DWT_d4Function.apply(x, self.wavelet)

class IWT(nn.Module):
    def __init__(self, wavelet='coif1'):
        super(IWT, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        return IWTFunction.apply(x, self.wavelet)


class DWT_d4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wavelet):
        B, C, H, W = x.shape
        coeffs_tensor = []
        dtype = x.dtype
        device = x.device

        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                # Extract the channel image
                img = x[b, c].detach().cpu().numpy()

                # Apply DWT
                coeffs2 = pywt.dwt2(img, wavelet)
                cA, (cH, cV, cD) = coeffs2

                # Convert to tensors and concatenate along the channel dimension
                cA_tensor = torch.tensor(cA, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cH_tensor = torch.tensor(cH, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cV_tensor = torch.tensor(cV, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
                cD_tensor = torch.tensor(cD, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)

                # Concatenate all coefficients
                combined = torch.cat((cA_tensor, cH_tensor, cV_tensor, cD_tensor), dim=0)  # Shape (4, H/2, W/2)
                batch_coeffs.append(combined)

            batch_coeffs_tensor = torch.cat(batch_coeffs, dim=0).unsqueeze(0)  # Shape (1, 4*C, H/2, W/2)
            coeffs_tensor.append(batch_coeffs_tensor)

        coeffs_tensor = torch.cat(coeffs_tensor, dim=0)  # Shape (B, 4*C, H/2, W/2)

        ctx.save_for_backward(x, coeffs_tensor)
        ctx.wavelet = wavelet

        return coeffs_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, coeffs_tensor = ctx.saved_tensors
        wavelet = ctx.wavelet

        B, C, H, W = x.shape
        C4 = grad_output.shape[1]
        C = C4 // 4
        H2, W2 = grad_output.shape[2], grad_output.shape[3]

        restored_grad = torch.zeros_like(x)

        for b in range(B):
            for c in range(C):
                # Extract the coefficients from the gradient tensor
                cA = grad_output[b, c * 4 + 0].cpu().numpy()
                cH = grad_output[b, c * 4 + 1].cpu().numpy()
                cV = grad_output[b, c * 4 + 2].cpu().numpy()
                cD = grad_output[b, c * 4 + 3].cpu().numpy()

                # Apply IWT
                restored_img = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
                restored_img_tensor = torch.tensor(restored_img, device=x.device, dtype=x.dtype)
                restored_grad[b, c] = restored_img_tensor

        return restored_grad, None




# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#
#         return dwt_init(x)
#
#
# class IWT(nn.Module):
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)


def dwt_init(x, wavelet='haar'):
    B, C, H, W = x.shape
    coeffs_tensor = []
    coeffs_LL = []
    dtype = x.dtype
    device = x.device

    for b in range(B):
        batch_coeffs = []
        batch_LL = []
        for c in range(C):
            # Extract the channel image
            img = x[b, c].detach().cpu().numpy()

            # Apply DWT
            coeffs2 = pywt.dwt2(img, wavelet)
            cA, (cH, cV, cD) = coeffs2

            # Convert to tensors and concatenate along the channel dimension
            cA_tensor = torch.tensor(cA, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cH_tensor = torch.tensor(cH, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cV_tensor = torch.tensor(cV, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cD_tensor = torch.tensor(cD, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)

            # Concatenate all coefficients
            combined = torch.cat((cA_tensor, cH_tensor, cV_tensor, cD_tensor), dim=0)  # Shape (4, H/2, W/2)
            batch_coeffs.append(combined)
            batch_LL.append(cA_tensor)

        batch_coeffs_tensor = torch.cat(batch_coeffs, dim=0)  # Shape (4*C, H/2, W/2)
        coeffs_tensor.append(batch_coeffs_tensor.unsqueeze(0))
        batch_LL_tensor = torch.cat(batch_LL, dim=0)  # Shape (4*C, H/2, W/2)
        coeffs_LL.append(batch_LL_tensor.unsqueeze(0))

    coeffs_tensor = torch.cat(coeffs_tensor, dim=0)
    coeffs_LL = torch.cat(coeffs_LL, dim=0)

    # x01 = x[:, :, 0::2, :] / 2
    # x02 = x[:, :, 1::2, :] / 2
    # x1 = x01[:, :, :, 0::2]
    # x2 = x02[:, :, :, 0::2]
    # x3 = x01[:, :, :, 1::2]
    # x4 = x02[:, :, :, 1::2]
    #
    # x_LL = x1 + x2 + x3 + x4
    # x_HL = -x1 - x2 + x3 + x4
    # x_LH = -x1 + x2 - x3 + x4
    # x_HH = x1 - x2 - x3 + x4
    # return  x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    return  coeffs_LL, coeffs_tensor

# class DWT_d4(nn.Module):
#     def __init__(self):
#         super(DWT_d4, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return dwt_init_d4(x)

def dwt_init_d4(x, wavelet='haar'):
    # x01 = x[:, :, 0::2, :] / 2
    # x02 = x[:, :, 1::2, :] / 2
    # x1 = x01[:, :, :, 0::2]
    # x2 = x02[:, :, :, 0::2]
    # x3 = x01[:, :, :, 1::2]
    # x4 = x02[:, :, :, 1::2]
    #
    # x_LL = x1 + x2 + x3 + x4
    # x_HL = -x1 - x2 + x3 + x4
    # x_LH = -x1 + x2 - x3 + x4
    # x_HH = x1 - x2 - x3 + x4
    # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    B, C, H, W = x.shape
    coeffs_tensor = []
    dtype = x.dtype
    device = x.device

    for b in range(B):
        batch_coeffs = []
        for c in range(C):
            # Extract the channel image
            img = x[b, c].detach().cpu().numpy()

            # Apply DWT
            coeffs2 = pywt.dwt2(img, wavelet)
            cA, (cH, cV, cD) = coeffs2

            # Convert to tensors and concatenate along the channel dimension
            cA_tensor = torch.tensor(cA, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cH_tensor = torch.tensor(cH, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cV_tensor = torch.tensor(cV, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)
            cD_tensor = torch.tensor(cD, device=device, dtype=dtype).unsqueeze(0)  # Shape (1, H/2, W/2)

            # Concatenate all coefficients
            combined = torch.cat((cA_tensor, cH_tensor, cV_tensor, cD_tensor), dim=0)  # Shape (4, H/2, W/2)
            batch_coeffs.append(combined)

        batch_coeffs_tensor = torch.cat(batch_coeffs, dim=0)  # Shape (4*C, H/2, W/2)
        coeffs_tensor.append(batch_coeffs_tensor.unsqueeze(0))

    coeffs_tensor = torch.cat(coeffs_tensor, dim=0)


    return   coeffs_tensor

def iwt_init(coeffs_tensor, wavelet='haar'):
    # r = 2
    # in_batch, in_channel, in_height, in_width = x.size()
    # # print([in_batch, in_channel, in_height, in_width])
    # out_batch, out_channel, out_height, out_width = in_batch, int(
    #     in_channel / (r ** 2)), r * in_height, r * in_width
    # x1 = x[:, 0:out_channel, :, :] / 2
    # x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    # x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    # x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    #
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    #
    # h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    # h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    # h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    # h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    B, C4, H2, W2 = coeffs_tensor.shape
    C = C4 // 4
    device = coeffs_tensor.device
    dtype = coeffs_tensor.dtype
    restored_images = []

    for b in range(B):
        batch_images = []
        for c in range(C):
            # Extract the coefficients
            cA = coeffs_tensor[b, c * 4 + 0].cpu().numpy()
            cH = coeffs_tensor[b, c * 4 + 1].cpu().numpy()
            cV = coeffs_tensor[b, c * 4 + 2].cpu().numpy()
            cD = coeffs_tensor[b, c * 4 + 3].cpu().numpy()

            # Apply IWT
            restored_img = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
            batch_images.append(torch.tensor(restored_img, device=device, dtype=dtype))

        restored_images.append(torch.stack(batch_images))

    return torch.stack(restored_images)


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
    
    
    
