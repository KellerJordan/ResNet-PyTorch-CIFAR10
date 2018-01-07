# PyTorch-ResNet-CIFAR10

This is a PyTorch implementation of Residual Networks as described in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Microsoft Research Asia. It is designed for the CIFAR-10 image classification task, following the ResNet architecture described on page 7 of the paper. This version allows use of dropout, arbitrary value of n, and a custom residual projection option.

#### Motivation

I completed this project in order to gain a better understanding of residual connections, which inspire the design of many state-of-the-art convnets at the present moment, as well as the gradient degradation problem. Having my own custom implementation made it easier to experiment with dropout and custom projection methods, and gave me practice with PyTorch.

## Usage

To train the network, use the following command:

```python main.py [-n=7] [--res-option='B'] [--use-dropout]```

### Default Hyperparameters

Hyperparameter | Default Value | Description
| - | - | - |
n | 5 | parameter controlling depth of network given structure described in paper
`res_option` | A | projection method when number of residual channels increases
`batch_size` | 128 | -
`weight_decay` | 0.0001 | -
`use_dropout` | False | -

## Results

Using `n=9` with otherwise default hyperparameters, the network achieves a test accuracy of 91.69%. This is somewhat lower than the result reported in the paper, likely because I used fewer training iterations due to compute limitations.
