import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 6, 3,padding=1)
        self.conv1_2 = nn.Conv2d(6, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv2_2 = nn.Conv2d(24, 36, 5)
        self.fc1 = nn.Linear(36 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool(F.relu(self.conv1_2(x)))
        x = F.relu(self.conv2_1(x))
        x = self.pool(F.relu(self.conv2_2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x