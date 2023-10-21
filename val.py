import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from models import CRNN
from utils import EncoderDecoder
import config

def val(img_list):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    model_name = 'CRNN_best.pth'

    en_decoder = EncoderDecoder(config.C2I_PATH, config.I2C_PATH)
    model = CRNN(2700).to(config.DEVICE) 
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH,model_name)))
    output_list = list()

    for img in img_list:

        input = val_transform(img)
        input = input.unsqueeze(0)
        input = input.to(config.DEVICE)
        outputs = model(input)
        output_str = en_decoder.TensorDecode(outputs[:,0,:])#一次一行，n=0
        output_list.append(output_str)

    return output_list