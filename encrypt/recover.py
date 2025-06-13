import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import zlib
from reedsolo import RSCodec
import os
print("当前工作目录：", os.getcwd())

# 从utils.py提取必要函数
class DWT:
    def __init__(self):
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

class IWT:
    def __init__(self):
        self.requires_grad = False

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

def bytearray_to_text(x):
    try:
        rs = RSCodec(128)
        text = rs.decode(x)
        text = zlib.decompress(text[0])
        return text.decode("utf-8")
    except:
        return False

def bits_to_bytearray(bits):
    ints = []
    bits = np.array(bits)
    bits = 0 + bits
    bits = bits.tolist()
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)

# 从block.py提取并修改INV_block - 修复通道问题
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

# 从d3net.py修改D3net - 使用原始键名结构
class D3net(nn.Module):
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

# 图像预处理
def transform2tensor(img_path, device):
    transform = T.Compose([
        T.CenterCrop((450, 600)),
        T.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def decode_steg(steg_path, model_path, device):
    # 禁用梯度计算以节省内存
    torch.set_grad_enabled(False)
    
    # 加载模型
    d3net = D3net()
    state_dicts = torch.load(model_path, map_location=device)
    
    # 修复键名不匹配问题
    network_state_dict = {}
    for k, v in state_dicts['net'].items():
        # 移除"model."前缀
        if k.startswith('model.'):
            new_key = k[6:]  # 移除前6个字符("model.")
        else:
            new_key = k
        network_state_dict[new_key] = v
    
    # 加载修正后的状态字典
    d3net.load_state_dict(network_state_dict)
    d3net.eval().to(device)
    
    # 加载隐写图像
    steg = transform2tensor(steg_path, device)
    print(f"Loaded steg image: {steg.shape}")
    
    # 小波变换
    dwt = DWT()
    steg_dwt = dwt.forward(steg)
    print(f"After DWT: {steg_dwt.shape}")
    
    # 创建一个与steg_dwt相同形状的零张量，作为负载部分的初始估计
    zeros = torch.zeros_like(steg_dwt)
    # 将封面图像的DWT特征与零张量拼接，形成24通道
    input_to_net = torch.cat([steg_dwt, zeros], dim=1)
    print(f"Input to network: {input_to_net.shape}")
    
    # 可逆网络反向传播恢复负载
    recovered = d3net.inverse(input_to_net)
    print(f"After inverse network: {recovered.shape}")
    
    # 提取后12通道作为负载的DWT特征
    payload_dwt = recovered.narrow(1, 12, 12)
    print(f"Payload DWT: {payload_dwt.shape}")
    
    # 逆小波变换恢复二值图像
    iwt = IWT()
    payload = iwt.forward(payload_dwt)
    print(f"After IWT: {payload.shape}")
    
    # 二值化处理
    binary_payload = (payload > 0.5).float().squeeze(0)
    bits = binary_payload.detach().cpu().numpy().flatten().astype(int).tolist()
    print(f"Extracted bits: {len(bits)}")
    
    # 比特流转文本
    candidates = {}
    byte_array = bits_to_bytearray(bits)
    print(f"Byte array length: {len(byte_array)}")
    text = bytearray_to_text(binary_payload)

    import re
    # 假设 byte_array 是字节数组
    decoded_texts = re.findall(b'[^\x00]+', byte_array)  # 匹配非零字节的序列
    candidates = {}
    for text in decoded_texts:
        text_str = bytearray_to_text(text)
        if text_str:
            candidates[text_str] = candidates.get(text_str, 0) + 1
            
    # 返回最可能的候选
    if not candidates:
        return "Flag not found"
    return max(candidates.items(), key=lambda x: x[1])[0]

if __name__ == '__main__':
    # 智能设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    steg_image = "./steg.png"
    model_weights = './misuha.taki'
    
    print(f"Decoding steg image: {steg_image}")
    print(f"Using model weights: {model_weights}")
    
    flag = decode_steg(steg_image, model_weights, device)
    print(f"\nExtracted Flag: {flag}")