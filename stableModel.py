import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.modules.activation import Softmax
from collections import OrderedDict

class StableModel(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1)),
            ("batchnorm1", nn.BatchNorm1d(num_features=512)),
            ("activation1", nn.LeakyReLU(0.01,inplace=False)),

            ("conv2", nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)),
            ("batchnorm2", nn.BatchNorm1d(num_features=256)),
            ("activation2", nn.LeakyReLU(0.01,inplace=False)),

            ("flatten", nn.Flatten()),
            # nn.Linear(256, 256), 
            # nn.LeakyReLU(),            
            ("linear1", nn.Linear(256,output_dim))
        ]))

    def forward(self,x):
        return self.model(x)
