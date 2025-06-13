import torch
from model import Model
from utils import DWT, IWT, make_payload, auxiliary_variable, bits_to_bytearray, bytearray_to_text
import torchvision
from PIL import Image
import torchvision.transforms as T

# 调整 transform_test，去掉 CenterCrop
transform_test = T.Compose([
    T.ToTensor(),
])

def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    simple_net.load_state_dict(network_state_dict)

def transform2tensor(img):
    img = Image.open(img)
    img = img.convert('RGB')
    img = img.resize((600, 450))  # 调整为 256x256
    return transform_test(img).unsqueeze(0).to(device)

def encode(cover, text):
    cover = transform2tensor(cover)
    B, C, H, W = cover.size()       
    payload = make_payload(W, H, C, text, B)
    payload = payload.to(device)
    cover_input = dwt(cover)
    payload_input = dwt(payload)        
    input_img = torch.cat([cover_input, payload_input], dim=1)

    with torch.no_grad():  # 关闭梯度计算
        output = simple_net(input_img)
    
    del input_img  # 删除中间张量
    torch.cuda.empty_cache()  # 清理缓存

    output_steg = output.narrow(1, 0, 4 * 3)
    output_img = iwt(output_steg)
    torchvision.utils.save_image(output_img, './steg.png')

if __name__ == '__main__':
    simple_net = Model()
    load('misuha.taki')
    simple_net.eval()

    dwt = DWT()
    iwt = IWT()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    text = r'flag{what_is_your_name}?'
    steg = r'./steg.png'
    cover = './yourname.png'
    encode(cover, text)