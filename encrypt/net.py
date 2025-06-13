from model import *
from block import INV_block


class simple_net(nn.Module):

    def __init__(self):
        super(simple_net, self).__init__()
        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

    def forward(self, x):

        out = self.inv1(x)
        out = self.inv2(out)
        out = self.inv3(out)
        out = self.inv4(out)
        out = self.inv5(out)
        out = self.inv6(out)
        out = self.inv7(out)
        out = self.inv8(out)
        return out


