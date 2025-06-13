from reblock import INV_block
import torch.nn as nn 

class simple_net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个块的输入通道数为24
        self.inv1 = INV_block(input_channels=24)
        self.inv2 = INV_block(input_channels=24)
        self.inv3 = INV_block(input_channels=24)
        self.inv4 = INV_block(input_channels=24)
        self.inv5 = INV_block(input_channels=24)
        self.inv6 = INV_block(input_channels=24)
        self.inv7 = INV_block(input_channels=24)
        self.inv8 = INV_block(input_channels=24)
    
    def inverse(self, x):
        # 逆序执行逆变换
        x = self.inv8.inverse(x)
        x = self.inv7.inverse(x)
        x = self.inv6.inverse(x)
        x = self.inv5.inverse(x)
        x = self.inv4.inverse(x)
        x = self.inv3.inverse(x)
        x = self.inv2.inverse(x)
        x = self.inv1.inverse(x)
        return x