import time
from utils.meters import AverageMeter
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms


def _load_training(root_path, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([128, 128]),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([128, 128]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader


def train_with_classifier(model, train_loader_dir, epoch=0, learning_rate=1e-4):
    t1 = time.time()
    model.train()

    kwargs = {'num_workers': 1, 'pin_memory': True}
    trainloader = _load_training(train_loader_dir, 128, kwargs)

    loss_func = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    step_count = len(trainloader)
    for step, (_x, _y) in enumerate(trainloader):
        _x = Variable(_x)
        _y = Variable(_y)
        _x, _y = _x.cuda(), _y.cuda()

        output = model(_x)
        loss = loss_func(output, _y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            pred_y = torch.max(output, 1)[1]
            acc = float((pred_y == _y).sum().item()) / _y.size(0)
            print('Epoch: %d(%.1f), loss: %.6f, accuracy: %2.2f[%d/%d]' % (epoch,
                                                                            ((step + 1) / step_count) * 100,
                                                                            loss.item(),
                                                                            acc, (pred_y == _y).sum(), _y.size(0)))

    print('Time token in epoch: %.1f s' % (time.time() - t1))
