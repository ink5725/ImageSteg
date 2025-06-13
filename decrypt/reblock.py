import torch 
import torch.nn as nn
from utils import initialize_weights

class ResidualDenseBlock_out(nn.Module):
    def __init__(self, channel=12, hidden_size=32, bias=True):
        super(ResidualDenseBlock_out, self).__init__()     
        self.channel = channel
        self.hidden_size = hidden_size   
        self.conv1 = nn.Conv2d(self.channel, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(self.channel + self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(self.channel + 2 * self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(self.channel + 3 * self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(self.channel + 4 * self.hidden_size, self.channel, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class INV_block(nn.Module):
    def __init__(self, input_channels=24, clamp=2.0):
        super().__init__()
        self.input_channels = input_channels
        self.clamp = clamp
        # 每个部分应该是输入通道数的一半
        self.split_channels = input_channels // 2
        self.r = ResidualDenseBlock_out(channel=self.split_channels)
        self.y = ResidualDenseBlock_out(channel=self.split_channels)
        self.f = ResidualDenseBlock_out(channel=self.split_channels)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))
    
    def inverse(self, y):
        # 将输入分成两个相等部分
        y1, y2 = (y.narrow(1, 0, self.split_channels),
                  y.narrow(1, self.split_channels, self.split_channels))
        
        s1 = self.r(y1)
        t1 = self.y(y1)
        e_s1 = self.e(s1)
        x2 = (y2 - t1) / e_s1
        t2 = self.f(x2)
        x1 = y1 - t2
        return torch.cat((x1, x2), 1)