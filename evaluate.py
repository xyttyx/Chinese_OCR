import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from datasets import train_Dataset,collate_fn
from torch.utils.data import DataLoader

from models import CRNN
from utils import EncoderDecoder, editDistance
import config
import time

def evaluate(model=None):
    
    en_decoder = EncoderDecoder(config.C2I_PATH, config.I2C_PATH)
    '''
    with open(config.TRAIN_LOG,'r') as log:
        _ , model_name = log.read().split()
    '''
    dataset = train_Dataset(config.TEST_IMG_PATH, config.TEST_LABEL_PATH)
    dataloader = DataLoader(dataset, batch_size = 48, shuffle=True, collate_fn=collate_fn,drop_last=True)
    
    if model is None:
        model_name = 'CRNN_epoch35.pth'
        model = CRNN(3000).to(config.DEVICE) # 与训练集保持一致
        model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, model_name)))

    accuracy_list = list()
    infer_time_list = list()
    for inputs, labels, _ in tqdm(dataloader,'evaluate processing') :

        #-------#
        begin_time = time.process_time_ns() 
        inputs = inputs.to(config.DEVICE)
        outputs = model(inputs)
        end_time = time.process_time_ns() 
        #-------#
        for i in range(len(inputs)):
            target = en_decoder.StringDecode(labels[i]).replace('#','')
            output_str = en_decoder.TensorDecode(outputs[:,i,:])
            edit_distance = editDistance(target, output_str)
            accuracy_list.append(1 - edit_distance / len(target))
            infer_time_list.append(end_time-begin_time)

    accuracy = sum(accuracy_list)/len(accuracy_list)
    infer_time_ns = sum(infer_time_list)/len(infer_time_list)
    print(f'Accuracy: {accuracy}, average infer time: {infer_time_ns/10000000}ms')
    return

if __name__ == '__main__':
    evaluate()