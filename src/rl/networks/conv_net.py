import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, state_dim, action_dim, alpha):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(384, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, 384))
        x = self.fc1(x)
        return x