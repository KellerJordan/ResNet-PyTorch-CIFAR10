# PyTorch-ResNet-CIFAR10

This is a PyTorch implementation of Residual Networks as described in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Microsoft Research Asia. It follows the description of ResNet configuration on page 7 of the paper for the CIFAR-10 image classification task, and allows any value of _n_(determining the number of layers) to be entered from the command line.

## Usage

To train the network, use the following command:

```python main.py [-n=7] [--res-option='B'] [--use-dropout]```

### Default Hyperparameters

Hyperparameter | Default Value | Description
- | -
`n` | 5 | parameter controlling depth of network given structure described in paper
`res_option` | `A` | projection method when number of residual channels increases
`batch_size` | 128 | -
`weight_decay` | 0.0001 | -
`use_dropout` | `False` | -

## Results

Using `n=9` with the residual connection type (A) and no dropout, the network achieves a test accuracy of .
