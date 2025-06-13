import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import zlib
from reedsolo import RSCodec
from utils import DWT, IWT, bits_to_bytearray, bytearray_to_text
from renet import simple_net

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
    d3net = simple_net()
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
    
    steg_image = "./decrypt/steg.png"
    model_weights = './encrypt/misuha.taki'
    
    print(f"Decoding steg image: {steg_image}")
    print(f"Using model weights: {model_weights}")
    
    flag = decode_steg(steg_image, model_weights, device)
    print(f"\nExtracted Flag: {flag}")