import argparse
import math
import os
import random
import shutil
import sys
import time
from os import path
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from compressai.models.ModNet import *


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        MSE = self.mse(output["x_hat"], target).item()
        out["PSNR"] = -10 * math.log(MSE, 10)

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_epoch(test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            psnr.update(out_criterion["PSNR"])

    print(
        f"\tPSNR loss: {psnr.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
    )

    return bpp_loss.avg, psnr.avg


def get_base_line():
    data_dir = "assets"
    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_dataset = ImageFolder(data_dir, split="Kodak24", transform=test_transforms)
    device = "cuda"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    for net in models:
        bpps = []
        psnrs = []
        ql = 6 if net[:2] == "ch" else 8
        for quality in range(1, ql + 1):
            model = models[net](quality, pretrained=True).cuda()
            bpp, psnr = test_epoch(test_dataloader, model, RateDistortionLoss())
            bpps.append(bpp)
            psnrs.append(psnr)
        plt.plot(bpps, psnrs, label=net)
    plt.legend()
    plt.xlabel("bpp")
    plt.ylabel("mse")
    plt.show()

def eval_mod():
    pass

def main():
    get_base_line()

if __name__ == "__main__":
    main()