import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module("linear1", nn.Linear(13, 70))
        self.sequential.add_module("relu1", nn.ReLU())
        #self.sequential.add_module("linear2", nn.Linear(30, 50))
        #self.sequential.add_module("relu2", nn.ReLU())
        #.sequential.add_module("linear3", nn.Linear(50, 30))
        #self.sequential.add_module("relu3", nn.ReLU())
        self.sequential.add_module("linear4", nn.Linear(70, 2))
        self.sequential.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, x) -> torch.Tensor:
        x = self.sequential(x.float())
        return x