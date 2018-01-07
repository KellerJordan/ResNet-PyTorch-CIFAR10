import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

from model import ResNet

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./dataset', type=str,
                    help='path to dataset')
parser.add_argument('--weight-decay', default=0.0001, type=float,
                    help='parameter to decay weights')
parser.add_argument('--num-epochs', default=10, type=int,
                    help='number of epochs to train for')
parser.add_argument('--batch-size', default=128, type=int,
                    help='size of each batch of cifar-10 training images')
parser.add_argument('--print-every', default=100, type=int,
                    help='number of iterations to wait before printing')

def main(args):
    if not torch.cuda.is_available():
        print('Error: CUDA library not available on system')
        return

    gpu_dtype = torch.cuda.FloatTensor

    # load model
    model = ResNet(5)
    model = model.type(gpu_dtype)
    param_count = get_param_count(model)
    print('Param count: %d' % param_count)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)

    # get CIFAR-10 data
    NUM_TRAIN = 45000
    NUM_VAL = 5000

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor()])

    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                                 transform=train_transform)
    loader_train = DataLoader(cifar10_train, batch_size=args.batch_size,
                              sampler=ChunkSampler(NUM_TRAIN))
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                               transform=T.ToTensor())
    loader_val = DataLoader(cifar10_train, batch_size=args.batch_size,
                            sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True,
                                transform=T.ToTensor())
    loader_test = DataLoader(cifar10_test, batch_size=args.batch_size)

    # train model
    for epoch in range(args.num_epochs):
        check_accuracy(model, loader_val)
        train(loader_train, model, criterion, optimizer, epoch)

    print('Final validation accuracy:')
    check_accuracy(model, loader_val)

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for X, y in loader:
        X_var = Variable(X.type(gpu_dtype), volatile=True)

        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(loader_train, model, criterion, optimizer, epoch):
    print('Starting epoch %d / %d' % (epoch+1, args.num_epochs))
    model.train()
    for t, (X, y) in enumerate(loader_train):
        X_var = Variable(X.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype))

        scores = model(X_var)

        loss = criterion(scores, y_var)
        if (t+1) % args.print_every == 0:
            print('t = %d, loss = %.4f' % (t+1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_param_count(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    return sum(param_counts)

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
