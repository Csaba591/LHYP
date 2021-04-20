import torch
from torch import nn
import torch.nn.functional as F
from ml_utils import conv_output_shape

class Linear(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(Linear, self).__init__()
        self.name = name
        
        h, w = input_shape
        self.fc1 = nn.Linear(in_features=num_channels*h*w, out_features=1)
    
    def forward(self, x):
        # start_dim = 1: don't flatten batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'

class SimpleCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(SimpleCNN, self).__init__()
        self.name = name
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=4)
        c1h, c1w = conv_output_shape(*input_shape, 4)
        self.fc1 = nn.Linear(in_features=12*c1h*c1w, out_features=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        # start_dim = 1: don't flatten batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'
    
class BasicCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(BasicCNN, self).__init__()
        self.name = name
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=12, kernel_size=4)
        c1_hw = conv_output_shape(*input_shape, 4)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=4)
        c2_hw = conv_output_shape(*c1_hw, 4)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3)
        
        conv_out_h, conv_out_w = conv_output_shape(*c2_hw, 3)
        self.fc1 = nn.Linear(in_features=6*conv_out_h*conv_out_w, out_features=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        # start_dim = 1: don't flatten batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'
    