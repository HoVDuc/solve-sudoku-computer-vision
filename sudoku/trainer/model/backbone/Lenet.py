import torch
import torch.nn as nn
from torch.functional import F


class CNN(nn.Module):

    def __init__(self, num_classes) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout2d()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.dropout1(out)
        out = self.maxpool3(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.bn5(out)
        out = self.maxpool4(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return F.softmax(out, -1)
