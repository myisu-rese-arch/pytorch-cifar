'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os, sys
import argparse

from models import *
from utils import progress_bar
import config as config

import torch as ch
import time
from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from tqdm import tqdm
from  torch.cuda.amp import autocast
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if config.dataset == 'CIFAR10':
    root = './data_cifar10'
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True)
elif config.dataset == 'CIFAR100':
    root = './data_cifar100'
    trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True)

    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True)
else:
    root = './data_imagenet'
    trainset = torchvision.datasets.ImageNet(
        root=root, train=True, download=True)

    testset = torchvision.datasets.ImageNet(
        root=root, train=False, download=True)

datasets = {
	'train': trainset,
        'test': testset
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'./data/{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

# Note that statistics are wrt to uin8 range, [0,255].
# CIFAR_MEAN = [125.307, 122.961, 113.8575]
# CIFAR_STD = [51.5865, 50.847, 51.255]

if config.dataset == 'CIFAR10':
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
elif config.dataset == 'CIFAR100':
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
else:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

MEAN = [255 * x for x in MEAN]
STD = [255 * x for x in STD]

BATCH_SIZE = config.batch_size

loaders = {}

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for name in ['train', 'test']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    if name == 'train':
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            #Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
        ])
    image_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(MEAN, STD),
    ])

    # Create loaders
    loaders[name] = Loader(f'./data/{name}.beton',
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            order=OrderOption.RANDOM,
                            drop_last=(name == 'train'),
                            pipelines={'image': image_pipeline,
                                       'label': label_pipeline})


trainloader = loaders['train']
testloader = loaders['test']

f = open('main.txt', 'w')
for i in range(config.trials):
    # Model
    best_acc = 0  # best test accuracy
    print('==> Building model..')
    net = ResNet18()
    net = net.to(memory_format=torch.channels_last).cuda()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    starttime = time.time()
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
    print(f"Training finished at: {(time.time() - starttime)} with accuracy: {best_acc}", f)
f.close()
