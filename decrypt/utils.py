import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
import math
from reedsolo import RSCodec
import zlib

rs = RSCodec(128)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    
def random_data(cover,device):
    return torch.zeros(cover.size(), device=device).random_(0, 2)

def auxiliary_variable(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def make_payload(width, height, depth, text, batch = 1):
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < batch * width * height * depth:
        payload += message

    
    payload = payload[:batch * width * height * depth]
    return torch.FloatTensor(payload).view(batch, depth, height, width)

def text_to_bits(text):
    return bytearray_to_bits(text_to_bytearray(text))

def bytearray_to_bits(x):
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result

def text_to_bytearray(text):
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x

def bits_to_bytearray(bits):
    ints = []
    bits = np.array(bits)
    bits = 0 + bits
    bits = bits = bits.tolist()
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)

def bytearray_to_text(x):
    try:
        text = rs.decode(x)
        text = zlib.decompress(text[0])
            
        return text.decode("utf-8")
    except BaseException:
        return False
