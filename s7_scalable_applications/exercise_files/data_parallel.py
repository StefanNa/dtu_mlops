from fashion_trainer import FashionCNN
import torch
import torch.nn as nn

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from torch import nn
model = MyModelClass()
model = nn.DataParallel(FashionCNN, device_ids=[0, 1])  # data parallel on gpu 0 and 1


train_set = FashionMNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=100)

dataiter = iter(dataloader)
batch, labels = dataiter.next()

import time
n_reps=2
start = time.time()
for _ in range(n_reps):
   out = model(batch)
end = time.time()

