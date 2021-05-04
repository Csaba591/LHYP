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

class MaxpoolCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(MaxpoolCNN, self).__init__()
        self.name = name
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=4)
        c1_hw = conv_output_shape(*input_shape, 4)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        mp1_hw = c1_hw[0] // 2, c1_hw[1] // 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4)
        c2_hw = conv_output_shape(*mp1_hw, 4)
        mp2_hw = c2_hw[0] // 2, c2_hw[1] // 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        
        conv_out_h, conv_out_w = conv_output_shape(*mp2_hw, 3)
        self.fc1 = nn.Linear(in_features=32*conv_out_h*conv_out_w, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        # start_dim = 1: don't flatten batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'
    
class DropoutCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(DropoutCNN, self).__init__()
        self.name = name
        
        self.do1 = nn.Dropout(p=0.25)
        self.do2 = nn.Dropout(p=0.25)

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=4)
        c1_hw = conv_output_shape(*input_shape, 4)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        mp1_hw = c1_hw[0] // 2, c1_hw[1] // 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4)
        c2_hw = conv_output_shape(*mp1_hw, 4)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        mp2_hw = c2_hw[0] // 2, c2_hw[1] // 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        
        conv_out_h, conv_out_w = conv_output_shape(*mp2_hw, 3)
        self.fc1 = nn.Linear(in_features=32*conv_out_h*conv_out_w, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = F.relu(x)

        x = self.do1(x)

        x = self.conv2(x)
        x = self.mp2(x)
        x = F.relu(x)
        
        x = self.do2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        # start_dim = 1: don't flatten batch dimension
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'
    
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False)

class Block(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        stride=1,
        activation=nn.ReLU(inplace=True),
        downsample=None):
        super(Block, self).__init__()
        
        self.activation = activation
        self.downsample = downsample

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        x = self.activation(x)

        return x
    
class ResNet(nn.Module):
    def __init__(
        self, 
        num_channels, 
        input_shape, 
        blocks,
        name, 
        activation=nn.ReLU(inplace=True)
    ):
        super(ResNet, self).__init__()
        self.name = name
        
        self.in_channels = 64

        channel_sizes = [64, 128, 256, 512]
        
        self.input_layer = nn.Conv2d(num_channels, self.in_channels, kernel_size=7, 
                                     stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(self.in_channels)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channel_sizes[0], blocks[0])
        self.layer2 = self._make_layer(channel_sizes[1], blocks[1], 2)
        self.layer3 = self._make_layer(channel_sizes[2], blocks[2], 2)
        self.layer4 = self._make_layer(channel_sizes[3], blocks[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_sizes[-1], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [Block(self.in_channels, out_channels, stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(Block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = torch.sigmoid(x)
        
        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'