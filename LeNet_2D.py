import torch.nn.functional as F

from torch import nn

class lenet_2D(nn.Module):
    def __init__(self):
        super(lenet_2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(41 * 41 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
