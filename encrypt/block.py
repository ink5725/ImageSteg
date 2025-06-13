import torch
import torch.nn as nn
from utils import initialize_weights

# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, bias=True):
        super(ResidualDenseBlock_out, self).__init__()     
        self.channel = 12
        self.hidden_size = 32   
        self.conv1 = nn.Conv2d(self.channel, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(self.channel + self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(self.channel + 2 * self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(self.channel + 3 * self.hidden_size, self.hidden_size, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(self.channel + 4 * self.hidden_size, self.channel, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class INV_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()
        
        self.channels = 3
        self.clamp = clamp
        # ρ
        self.r = ResidualDenseBlock_out()
        # η
        self.y = ResidualDenseBlock_out()
        # φ
        self.f = ResidualDenseBlock_out()

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x):
        x1, x2 = (x.narrow(1, 0, self.channels*4),
                  x.narrow(1, self.channels*4, self.channels*4))

        t2 = self.f(x2)
        y1 = x1 + t2
        s1, t1 = self.r(y1), self.y(y1)
        y2 = self.e(s1) * x2 + t1

        return torch.cat((y1, y2), 1)

