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
parser.add_argument('--batch-size', default=128, type=int,
                    help='size of each batch of cifar-10 training images')
parser.add_argument('--print-every', default=100, type=int,
                    help='number of iterations to wait before printing')
parser.add_argument('-n', default=5, type=int,
                    help='value of n to use for resnet configuration (see https://arxiv.org/pdf/1512.03385.pdf for details)')
parser.add_argument('--use-dropout', default=False, const=True, nargs='?',
                    help='whether to use dropout in network')
parser.add_argument('--res-option', default='A', type=str,
                    help='which projection method to use for changing number of channels in residual connections')

def main(args):
    # define transforms for normalization and data augmentation
    transform_augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4)])
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # get CIFAR-10 data
    NUM_TRAIN = 45000
    NUM_VAL = 5000
    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                                 transform=T.Compose([transform_augment, transform_normalize]))
    loader_train = DataLoader(cifar10_train, batch_size=args.batch_size,
                              sampler=ChunkSampler(NUM_TRAIN))
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                               transform=transform_normalize)
    loader_val = DataLoader(cifar10_train, batch_size=args.batch_size,
                            sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True,
                                transform=transform_normalize)
    loader_test = DataLoader(cifar10_test, batch_size=args.batch_size)
    
    # load model
    model = ResNet(args.n, res_option=args.res_option, use_dropout=args.use_dropout)
    
    param_count = get_param_count(model)
    print('Parameter count: %d' % param_count)
    
    # use gpu for training
    if not torch.cuda.is_available():
        print('Error: CUDA library unavailable on system')
        return
    global gpu_dtype
    gpu_dtype = torch.cuda.FloatTensor
    model = model.type(gpu_dtype)
    
    # setup loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # train model
    SCHEDULE_EPOCHS = [50, 5, 5] # divide lr by 10 after each number of epochs
#     SCHEDULE_EPOCHS = [100, 50, 50] # divide lr by 10 after each number of epochs
    learning_rate = 0.1
    for num_epochs in SCHEDULE_EPOCHS:
        print('Training for %d epochs with learning rate %f' % (num_epochs, learning_rate))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(num_epochs):
            check_accuracy(model, loader_val)
            print('Starting epoch %d / %d' % (epoch+1, num_epochs))
            train(loader_train, model, criterion, optimizer)
        learning_rate *= 0.1
    
    print('Final test accuracy:')
    check_accuracy(model, loader_test)

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

def train(loader_train, model, criterion, optimizer):
    model.train()
    for t, (X, y) in enumerate(loader_train):
        X_var = Variable(X.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype)).long()

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
