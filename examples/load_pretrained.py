import torch

from compressai.zoo import bmshj2018_hyperprior
from PIL import Image
from torchvision.transforms import ToTensor, RandomCrop
import numpy as np
from matplotlib import pyplot as plt

net = bmshj2018_hyperprior(8, pretrained=True).cuda()
t = ToTensor()(Image.open("../assets/3a6367b639451eb4ba17bd80447c5ba7.jpeg")).cuda()
t = RandomCrop((512, 512))(t)
print(t.size())
t = torch.reshape(t, (1, t.size()[0], t.size()[1], t.size()[2]))
print(t.size())
y = net.compress(t)
x_ = net.decompress(**y)["x_hat"][0]
x_ = x_.cpu().detach().numpy()
x_ = np.transpose(x_, [1, 2, 0])
plt.imshow(x_)
plt.show()