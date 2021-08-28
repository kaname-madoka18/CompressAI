import argparse
import math
import os
import random
import shutil
import sys
import time
from os import path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from compressai.models.ModNet import *

batchsize = 16
num_workers = 4
learning_rate = 1e-4
checkpoint = None

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
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
        out["mse_loss"] = sum(
            output["lambda"][i] * self.mse(output["x_hat"][i], target[i]) / N
            for i in range(N)
        )
        out["loss"] = 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

def adjust_learning_rate(optimizer, epoch, init_lr):

    if epoch < 10:
        lr = init_lr
    else:
        lr = init_lr * (0.5 ** ((epoch-7) // 3))
    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, out_dir
):
    model.train()
    device = next(model.parameters()).device
    last_time = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 10 == 0:
            t = time.time()
            time_used = t - last_time
            last_time = t
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f'\ttime used: {time_used:.2f}s |'
            )

        if i % 1000 == 0:
            torch.save(model.state_dict(), path.join(out_dir, "checkpoint_latest"))

def get_pretrained_scale():
    net = ScaleMod(128, 192).cuda()
    hyperprior = models["bmshj2018-hyperprior"](5, pretrained=True).cuda()
    net.g_a = hyperprior.g_a.cuda()
    net.g_s = hyperprior.g_s.cuda()
    net.h_a = hyperprior.h_a.cuda()
    net.h_s = hyperprior.h_s.cuda()
    return net, "checkpoint_scaled_hyper"

def get_pretrained_joint():
    net = JointMod(192, 320).cuda()
    joint = models["mbt2018"](5, pretrained=True).cuda()
    net.g_a = joint.g_a
    net.g_s = joint.g_s
    net.h_a = joint.h_a
    net.h_s = joint.h_s
    net.entropy_parameters = joint.entropy_parameters
    net.context_prediction = joint.context_prediction
    return net, "checkpoint_joint"

def get_pretrained_cheng_anchor():
    net = Cheng2020AnchorMod(192).cuda()
    anchor = models["cheng2020-anchor"](4, pretrained=True).cuda()
    net.g_a = anchor.g_a
    net.g_s = anchor.g_s
    net.h_a = anchor.h_a
    net.h_s = anchor.h_s
    net.entropy_parameters = anchor.entropy_parameters
    net.context_prediction = anchor.context_prediction
    return net, "checkpoint_cheng_anchor"

def get_pretrained_cheng_attention():
    net = Cheng2020AttentionMod(192).cuda()
    attention = models["cheng2020-attn"](4, pretrained=True).cuda()
    net.g_a = attention.g_a
    net.g_s = attention.g_s
    net.h_a = attention.h_a
    net.h_s = attention.h_s
    net.entropy_parameters = attention.entropy_parameters
    net.context_prediction = attention.context_prediction
    return net, "checkpoint_cheng_attention"

def test_models():
    get_pretrained_scale()
    get_pretrained_cheng_attention()
    get_pretrained_cheng_anchor()
    get_pretrained_joint()

def main():
    models2train = [get_pretrained_joint, get_pretrained_cheng_anchor, get_pretrained_cheng_attention]

    for factory in models2train:
        whole, out_dir = factory()
        if not path.isdir(out_dir):
            os.makedirs(out_dir)

        train_transforms = transforms.Compose(
            [transforms.RandomCrop((256, 256)), transforms.ToTensor()]
        )
        train_data = ImageFolder(path.join("..", "DS"), transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True)


        criterion = RateDistortionLoss()
        optimizer = torch.optim.Adam(whole.parameters(), lr = learning_rate)

        for epoch in range(1):
            cur_lr = adjust_learning_rate(optimizer, epoch, learning_rate)
            train_one_epoch(whole, criterion, train_loader, optimizer, epoch, 5, out_dir=out_dir)
            torch.save(whole.state_dict(), path.join(out_dir, f"checkpoint_epoch_{epoch}"))

if __name__ == "__main__":
    main()