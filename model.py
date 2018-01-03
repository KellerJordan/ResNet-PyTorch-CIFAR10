import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def get_model():
    model_base = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(5408, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 10),
    )

    return model_base
