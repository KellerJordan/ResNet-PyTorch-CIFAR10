import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
# import torchvision.transforms as T

# import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('data', default='./dataset', help='path to dataset')
parser.add_argument('weight_decay', default=0.0, help='parameter to decay weights')
parser.add_argument('num_epochs', default=10, help='number of epochs to train for')
parser.add_argument('print_every', default=100, help='number of iterations to wait before printing')

def main():
    global args
    args = parser.parse_args()
    gpu_dtype = torch.cuda.FloatTensor

    # load model
    # model = get_model()
    # model.cuda()
    # model = model.type(gpu_dtype)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(),
                           weight_decay=args.weight_decay)

    # get CIFAR-10 data
    NUM_TRAIN = 49000
    NUM_VAL = 1000
    NUM_TEST = 10000
    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True, transform=T.ToTensor())
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
    # cifar10_val = dset.CIFAR10('./dataset', train=True, download=True, transform=T.ToTensor())
    loader_val = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, NUM_VAL))
    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, transform=T.ToTensor())
    loader_test = DataLoader(cifar10_test, batch_size=64)

    # train model
    for epoch in range(args.num_epochs):
        train(loader_train, model, criterion, optimizer, epoch)
        

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

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
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

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples
