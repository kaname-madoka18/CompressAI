import os

from myTrain import *
from os import path
import math

def PSNR(x, x_):
    residual = x - x_
    mse = torch.linalg.norm((residual).view(residual.size[0], -1), ord=2).item()
    return 10*math.log(255*255/mse, 10)

def verify(test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    bpp = AverageMeter()
    distortion = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            compressed = model.compress(d)
            d_ = model.decompress(compressed)["x_hat"]
            distortion.update(criterion(d, d_))
            bpp.update(compressed)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    testfolder = "../../DS/test"
    net = ScaleHyperprior(128, 192)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    qualities = ["001", "005", "01", "02"]

    test_transforms = transforms.Compose(
        [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
    )

    test_dataset = ImageFolder(testfolder, transform=test_transforms)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        pin_memory=(device=="cuda")
    )

    for lam in qualities:
        dirpath = f"lam{lam}"
        assert path.isdir(dirpath)
        checkpoint = torch.load(path.join(dirpath, "checkpoint.pth.tar"), map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        verify(test_dataloader, net, PSNR)


if __name__ == "__main__":
    main()