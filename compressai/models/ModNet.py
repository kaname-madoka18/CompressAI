import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models import *

from .utils import conv, deconv, update_registered_buffers

class bm(nn.Module):
    def __init__(self):
        super(bm, self).__init__()

        self.fcn1 = nn.Linear(1, 100)
        self.fcn2 = nn.Linear(100, 100)
        self.fcn3 = nn.Linear(100, 100)
        self.Relu = nn.ReLU()

    def forward(self, y):
        y1 = self.fcn1(y)
        y1 = self.Relu(y1)
        y2=self.fcn2(y1)
        y2=self.Relu(y2)
        mask = 1-F.sigmoid(self.fcn3(y2))

        return mask

class Modnet(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Modnet, self).__init__()

        self.m = int(in_channels)
        self.c = int(latent_channels)
        self.conv1 = nn.Conv2d(self.m, self.c, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv4 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv5 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv6 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv7 = nn.Conv2d(self.c, self.c, 1, 1, 0)
        self.conv8 = nn.Conv2d(self.c, self.m, 1, 1, 0)

        self.lmbd_map1 = bm()
        self.lmbd_map2 = bm()
        self.lmbd_map3 = bm()
        self.lmbd_map4 = bm()
        self.lmbd_map5 = bm()
        self.lmbd_map6 = bm()
        self.lmbd_map7 = bm()

    def forward(self, x, lmd):

        b=x.size()[0]
        y=lmd.cuda()

        mask1 = self.lmbd_map1(y)
        mask1 = torch.reshape(mask1,(b,self.c,1,1)).cuda()
        mask2 = self.lmbd_map2(y)
        mask2 = torch.reshape(mask2, (b, self.c, 1, 1)).cuda()
        mask3 = self.lmbd_map3(y)
        mask3 = torch.reshape(mask3, (b, self.c, 1, 1)).cuda()
        mask4 = self.lmbd_map4(y)
        mask4 = torch.reshape(mask4, (b, self.c, 1, 1)).cuda()
        mask5 = self.lmbd_map5(y)
        mask5 = torch.reshape(mask5, (b, self.c, 1, 1)).cuda()
        mask6 = self.lmbd_map6(y)
        mask6 = torch.reshape(mask6, (b, self.c, 1, 1)).cuda()
        mask7 = self.lmbd_map7(y)
        mask7 = torch.reshape(mask7, (b, self.c, 1, 1)).cuda()

        x1 = x
        x1 = self.conv1(x1)
        x2 = mask1 * x1
        x2 = self.conv2(x2)
        x3 = mask2*x2
        x3 = self.conv3(x3)
        x4 = mask3*x3
        x4 = self.conv4(x4)
        x5 = mask4*x4
        x5 = self.conv5(x5)
        x6 = mask5*x5
        x6 = self.conv6(x6)
        x7 = mask6 * x6
        x7 = self.conv7(x7)
        x8 = mask7 * x7
        x8 = self.conv8(x8)

        output = x * (1 - F.sigmoid(x8))

        return output


class ScaleMod(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.modnet1 = Modnet(M, 100)
        self.modnet2 = Modnet(M, 100)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        b = x.size()[0]
        if self.training:
            lambda_info = torch.rand((b, 1), device="cuda") / 5
        else:
            lambda_info = torch.ones((b, 1), device="cuda") * self.lam

        y = self.g_a(x)
        y = self.modnet1(y, lambda_info)
        y = self.modnet2(y, lambda_info)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lambda": lambda_info
        }


class JointMod(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super(JointMod, self).__init__(N, M, **kwargs)
        self.modnet1 = Modnet(M, 100)
        self.modnet2 = Modnet(M, 100)

    def forward(self, x):
        b = x.size()[0]
        if self.training:
            lambda_info = torch.rand((b, 1), device="cuda") / 5
        else:
            lambda_info = torch.ones((b, 1), device="cuda") * self.lam

        y = self.g_a(x)
        y = self.modnet1(y, lambda_info)
        y = self.modnet2(y, lambda_info)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lambda": lambda_info
        }


class Cheng2020AnchorMod(Cheng2020Anchor):
    def __init__(self, N, **kwargs):
        super(Cheng2020AnchorMod, self).__init__(N, **kwargs)

        self.modnet1 = Modnet(N, 100)
        self.modnet2 = Modnet(N, 100)

    def forward(self, x):
        b = x.size()[0]
        if self.training:
            lambda_info = torch.rand((b, 1), device="cuda") / 5
        else:
            lambda_info = torch.ones((b, 1), device="cuda") * self.lam

        y = self.g_a(x)
        y = self.modnet1(y, lambda_info)
        y = self.modnet2(y, lambda_info)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lambda": lambda_info
        }


class Cheng2020AttentionMod(Cheng2020AnchorMod):
    def __init__(self, N, **kwargs):
        super(Cheng2020AttentionMod, self).__init__(N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
