import torch
import math
import torch.nn.functional as F
from torch import nn

class MLP_MNIST(torch.nn.Module):
    # MLP
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class CNN_MNIST(torch.nn.Module):
    # MLP
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3))
        self.conv2 = torch.nn.Conv2d(16, 32, (4, 4))
        self.fc1 = torch.nn.Linear(32 * 25, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))  # 16 * 13 * 13
        x = torch.max_pool2d(torch.relu(self.conv2(x)), (2, 2))  # 32 * 5 * 5
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class CNN_CIFAR10(torch.nn.Module):
    # MLP
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3))
        self.conv2 = torch.nn.Conv2d(16, 32, (4, 4))
        self.conv3 = torch.nn.Conv2d(32, 64, (3, 3))
        self.fc1 = torch.nn.Linear(64 * 2 * 2, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), (2, 2)))  # 64 * 15 * 15
        x = torch.relu(torch.max_pool2d(self.conv2(x), (2, 2)))  # 128 * 6 * 6
        x = torch.relu(torch.max_pool2d(self.conv3(x), (2, 2)))  # 256 * 2 * 2
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class MLP_CIFAR10(torch.nn.Module):
    # MLP
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class VGG(torch.nn.Module):
    '''
    VGG model
    '''

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out

class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        # 定义每一个就卷积层
        self.layer1 = nn.Sequential(
            # 卷积层  #输入图像为1*28*28#Mnist改为(1,32,,3,1)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层特征图通道数不改变，每个特征图的分辨率变小
            # 激活函数Relu
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # 定义全连接层
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # 对应十个类别的输出

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x




class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)
