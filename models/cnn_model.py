import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)

        self.pool = nn.MaxPool2d(2)

        # Adaptive pooling -> fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))

        self.fc = nn.Linear(64*4*4,128)

    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0),-1)

        return self.fc(x)