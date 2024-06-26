# adverarial training on cifar10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
from torchvision.models import resnet18

def adv_train(model, device, train_loader, optimizer, epoch, epsilon, loss_fn):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        model.to(device)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # perturbed_data = pgd_attack(model, loss_fn, target, data, epsilon)\
        perturbed_data = data
        perturbed_output = model(perturbed_data)
        adv_loss = loss_fn(perturbed_output, target)
        adv_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), adv_loss.item()))
    return model

def pgd_attack(model, loss_fn, target, data, epsilon, steps=10):
    data.requires_grad = True
    model.to(data.device)
    step_size = epsilon / 4

    perturbed_data = data.clone().detach()

    for _ in range(steps):
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = loss_fn(output, target)
        grad = torch.autograd.grad(loss, perturbed_data)[0]
        perturbed_data = perturbed_data + step_size * grad.sign()
        delta = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
        perturbed_data = torch.clamp(data + delta, min=0, max=1).detach()
    return perturbed_data

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def main(epsilon=8, batch_size=256, epochs=60,resume = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False).to(device)
    model.fc = nn.Linear(512, 10)
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
    epsilon = epsilon/255

    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])), batch_size=batch_size, shuffle=True)

    if resume:
        model.load_state_dict(torch.load(f'cifar10_resnet18_radius_{epsilon}.pth'))
    for epoch in range(1, epochs + 1):
        model = adv_train(model, device, train_loader, optimizer, epoch, epsilon, loss_fn)
        accuracy = test(model, device, test_loader, loss_fn)
        scheduler.step()
        torch.save(model.state_dict(), f'cifar10_resnet18_radius_{epsilon}.pth')

if __name__ == '__main__':
    main(epsilon=1, batch_size=100, epochs=200, resume=False)