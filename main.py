'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys, time
import argparse

from models import *
from utils import progress_bar
import config as config

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 1
    for item1, item2 in zip(trainloader1, trainloader2):
        inputs1, targets1 = item1
        inputs2, targets2 = item2
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs2, targets2 = inputs2.to(device), targets2.to(device)
        optimizer.zero_grad()
        outputs1 = net(inputs1)
        loss = criterion(outputs1, targets1)
        outputs2 = net(inputs2)
        loss += criterion(outputs2, targets2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs1.max(1)
        # total += targets1.size(0)

        _, pred1 = outputs1.max(1)
        total += targets1.size(0)

        _, pred2 = outputs2.max(1)
        total += targets2.size(0)

        # correct += predicted.eq(targets).sum().item()
        correct += pred1.eq(targets1).sum().item() + pred2.eq(targets2).sum().item()

        progress_bar(batch_idx, len(trainloader1), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        batch_idx = batch_idx + 1
    acc = 100.*correct/total
    train_loss /= len(trainloader1.dataset)

    return acc, train_loss

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
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
    test_loss /= len(testloader.dataset)
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

    return acc, test_loss


if config.dataset == 'CIFAR10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
elif config.dataset == 'CIFAR100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
else:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 1000

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])

transform_train_1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_train_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if config.dataset == 'CIFAR10':
    root = './data_cifar10'
    # trainset = torchvision.datasets.CIFAR10(
    #     root=root, train=True, download=True, transform=transform_train)
    trainset1 = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train_1)

    trainset2 = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train_2)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
elif config.dataset == 'CIFAR100':
    root = './data_cifar100'
    # trainset = torchvision.datasets.CIFAR100(
    #     root=root, train=True, download=True, transform=transform_train)

    trainset1 = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train_1)

    trainset2 = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train_2)

    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test)
else:
    root = './data_imagenet'
    # trainset = torchvision.datasets.ImageNet(
    #     root=root, train=True, download=True, transform=transform_train)

    trainset1 = torchvision.datasets.ImageNet(
        root=root, train=True, download=True, transform=transform_train_1)

    trainset2 = torchvision.datasets.ImageNet(
        root=root, train=True, download=True, transform=transform_train_2)

    testset = torchvision.datasets.ImageNet(
        root=root, train=False, download=True, transform=transform_test)

# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

trainloader1 = torch.utils.data.DataLoader(
    trainset1, batch_size=config.batch_size, shuffle=True, num_workers=2)

trainloader2 = torch.utils.data.DataLoader(
    trainset2, batch_size=config.batch_size, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

timings, accuracy = [], []
for i in range(config.trials):
    # Model
    best_acc = 0  # best test accuracy
    print('==> Building model..')
    # net = ResNet18(num_classes)
    net = SimpleDLA(num_classes=num_classes)
    net = net.to(device)
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
    train_acc, train_loss = [], []
    test_acc, test_loss = [], []
    for epoch in range(start_epoch, start_epoch+config.EPOCHS):
        tr_acc, tr_loss = train(epoch)
        train_acc.append(tr_acc)
        train_loss.append(tr_loss)

        te_acc, te_loss = test(epoch)
        test_acc.append(te_acc)
        test_loss.append(te_loss)

        scheduler.step()
    config.plot(config.EPOCHS, train_acc, test_acc, 'Accuracy', extra = config.output_name +  + str(i))
    config.plot(config.EPOCHS, train_loss, test_loss, 'Loss', extra = config.output_name +  + str(i))

    timing = time.time() - starttime
    timings.append(timing)
    accuracy.append(best_acc)

    print(f"Training finished at: {(timing)} with accuracy: {best_acc}")
    config.write_list_to_csv(config.trials, config.output_name + '_tim_acc', timings, accuracy)
