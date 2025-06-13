import torch.nn as nn
import torch
from net import simple_net


class Model(nn.Module):
    def __init__(self,cuda=True):
        super(Model, self).__init__()
        self.model = simple_net()
        if cuda:
            self.model.cuda()
        # init_model(self)

    def forward(self, x):
        out = self.model(x)
        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)