import torch.nn as nn
import torch.nn.functional as F

class RPS_net_mlp(nn.Module):

    def __init__(self):
        super(RPS_net_mlp, self).__init__()
        self.init()

    def init(self):
        """Initialize all parameters"""
        self.mlp1 = nn.Linear(784, 400)
        self.mlp2 = nn.Linear(400, 400)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(400, 10, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.cuda()

    def forward(self, x):
        x = x.view(-1, 784)
        y = self.mlp1(x)
        y = F.relu(y)

        y = self.mlp2(y)
        y = F.relu(y)

        x1 = self.fc(y)
        return x1