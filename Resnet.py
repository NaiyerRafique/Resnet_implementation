import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


# Create a residual block with two convolution layers and a skip connection with option B ( The paper uses Option A)
class residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=1):
        super(residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding=padding)
        self.skipconn = lambda x: x
        if stride != 1 or in_channels != out_channels:
            self.skipconn = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        residual = self.skipconn(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = out + residual
        return out


# Use the residual block in the as the building block of the overall network

class resnet(nn.Module):
    def __init__(self, residual_block, num_layers):
        super(resnet, self).__init__()
        self.layers = []
        self.in_channels = 16
        self.conv3x3 = nn.Conv2d(3, 16, 3, 1, 1)
        self.residual_block1 = self.layer_for_block1(residual_block, num_layers[0], 16, initial_stride=1)
        self.residual_block2 = self.layer_for_block2(residual_block, num_layers[1], 32, initial_stride=2)
        self.residual_block3 = self.layer_for_block3(residual_block, num_layers[2], 64, initial_stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64, 10)

# First block creates a feature space of size 16x32x32
    def layer_for_block1(self, residual_block, num_layers, out_channels, initial_stride):
        stride_for_layer1 = [initial_stride] + [1]*(num_layers - 1)
        for stride in stride_for_layer1:
            res_block1 = residual_block(self.in_channels, out_channels, stride)
            #self.in_channels = out_channels
            self.layers.append(res_block1)
        return self.layers

# Second block creates a feature space of size 32x16x16
    def layer_for_block2(self, residual_block, num_layers, out_channels, initial_stride):
        stride_for_layer2 = [initial_stride] + [1]*(num_layers - 1)
        for stride in stride_for_layer2:
            res_block2 = residual_block(self.in_channels, out_channels, stride)
            self.in_channels = out_channels
            self.layers.append(res_block2)
        return self.layers
        #return nn.Sequential(*self.layers)

# Third block creates a feature space of size 64x8x8
    def layer_for_block3(self, residual_block, num_layers, out_channels, initial_stride):
        stride_for_layer3 = [initial_stride] + [1]*(num_layers - 1)
        for stride in stride_for_layer3:
            res_block3 = residual_block(self.in_channels, out_channels, stride)
            self.in_channels = out_channels
            self.layers.append(res_block3)
        #return self.layers
        return nn.Sequential(*self.layers)



    def forward(self, x):
        out = self.conv3x3(x)
        out = self.residual_block3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out




model = resnet(residual, [3, 3, 3])


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))