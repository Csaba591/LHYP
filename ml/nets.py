import torch
from torch import nn
import torch.nn.functional as F
from ml_utils import conv_output_shape
from torchvision.models import resnet18, resnet34, resnet101

class DropoutCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name):
        super(DropoutCNN, self).__init__()
        self.name = name

        self.do1 = nn.Dropout(p=0.25)
        self.do2 = nn.Dropout(p=0.25)

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=64, kernel_size=4)
        c1_hw = conv_output_shape(*input_shape, 4)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        mp1_hw = c1_hw[0] // 2, c1_hw[1] // 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4)
        c2_hw = conv_output_shape(*mp1_hw, 4)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        mp2_hw = c2_hw[0] // 2, c2_hw[1] // 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)

        conv_out_h, conv_out_w = conv_output_shape(*mp2_hw, 3)
        self.fc1 = nn.Linear(in_features=32*conv_out_h *
                             conv_out_w, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

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
        activation=nn.ReLU(inplace=True),
        use_dropout=False
    ):
        super(ResNet, self).__init__()
        self.name = name
        self.use_dropout = use_dropout

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

        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
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

        layers = [Block(self.in_channels, out_channels,
                        stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        if self.use_dropout: x = self.do1(x)

        x = self.layer1(x)
        if self.use_dropout: x = self.do1(x)
        
        x = self.layer2(x)
        if self.use_dropout: x = self.do1(x)
        
        x = self.layer3(x)
        if self.use_dropout: x = self.do1(x)
        
        x = self.layer4(x)
        if self.use_dropout: x = self.do2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'


class LongCNN(nn.Module):
    def __init__(self, num_channels, input_shape, name, conv_sizes=[64, 128, 128, 256], lin_size=512):
        super(LongCNN, self).__init__()
        self.name = name

        self.relu = nn.ReLU(inplace=True)

        self.do1 = nn.Dropout(p=0.25)
        self.do2 = nn.Dropout(p=0.25)

        self.conv1 = nn.Conv2d(num_channels, out_channels=conv_sizes[0], 
            kernel_size=6, stride=3)
        
        self.conv2 = nn.Conv2d(
            in_channels=conv_sizes[0], out_channels=conv_sizes[1], 
            kernel_size=4, stride=2)
       
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(
            in_channels=conv_sizes[1], out_channels=conv_sizes[2], kernel_size=3)
        
        self.conv4 = nn.Conv2d(
            in_channels=conv_sizes[2], out_channels=conv_sizes[3], kernel_size=3)
        
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        #self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(in_features=conv_sizes[-1], out_features=lin_size)
        self.fc2 = nn.Linear(in_features=lin_size, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu(x)

        # x = self.do1(x)

        x = self.conv3(x)
        x = self.pool1(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.pool1(x)
        x = self.relu(x)

        # x = self.do2(x)

        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'

class SimmCNN(nn.Module): 
    def __init__(self, num_channels, input_shape, name):
        super(SimmCNN, self).__init__() 

        self.name = name

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(num_channels, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x): 
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = torch.flatten(x, start_dim=1) 
    
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    @property
    def save_path(self):
        return self.name + '.chkpt'


def create_resnet(resnet_fn, in_channels, expansion, name):
    rn = resnet_fn()
    rn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                         stride=2, padding=3, bias=False)
    # rn.conv1 = nn.Conv2d(
    #         in_channels, 64, kernel_size=4, stride=2, padding=2, bias=False)
    rn.fc = nn.Linear(512 * expansion, 1)
    rn.name = name
    rn.save_path = name + '.chkpt'
    return rn


def create_model(model_type, in_channels, input_shape, name, variant):
    mt = model_type.lower()
    var = variant.lower()
    if mt == 'resnet18':
        if var == 'dropout':
            return ResNet(
                in_channels,
                input_shape,
                [2, 2, 2, 2],
                name,
                use_dropout=True
            )
        return create_resnet(resnet18, in_channels, 1, name)
        # return nets.ResNet(in_channels, input_shape, [2,2,2,2], name)
    elif mt == 'resnet34':
        if var == 'dropout':
            return ResNet(
                in_channels,
                input_shape,
                [3, 4, 6, 3],
                name,
                use_dropout=True
            )
        elif var == 'pytorch':
            return create_resnet(resnet34, in_channels, 1, name)
        return ResNet(in_channels, input_shape, [3, 4, 6, 3], name)
    elif mt == 'resnet101':
        return create_resnet(resnet101, in_channels, 4, name)
    elif mt == 'dropoutcnn':
        return DropoutCNN(in_channels, input_shape, name)
    elif mt == 'longcnn':
        return LongCNN(in_channels, input_shape, name)
    elif mt == 'simmcnn':
        return SimmCNN(in_channels, input_shape, name)
    else:
        raise ValueError(f'Unknown model type \"{model_type}\"')
