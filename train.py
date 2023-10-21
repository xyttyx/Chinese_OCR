import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from datasets import train_Dataset, collate_fn
from torch.utils.data import DataLoader

from models import CRNN
from utils import EncoderDecoder
from evaluate import evaluate
import config

def train():
    '''   
    with open(config.TRAIN_LOG,'r') as log:
        current_epoch, model_name = log.read().split()
        current_epoch = int(current_epoch)
    '''
    model_name = 'CRNN_epoch86.pth'
    current_epoch = 86
    dataset = train_Dataset(config.TRAIN_IMG_PATH, config.TRAIN_LABEL_PATH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = CRNN(2700)
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, model_name))) 
    model = model.to(config.DEVICE)

    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5)
    
    for epoch in range(current_epoch + 1, current_epoch + 1 + config.TRAIN_EPOCH):
        for inputs, labels, len_labels in tqdm(dataloader,f'epoch:{epoch}'):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            len_labels = len_labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            len_seq = torch.ones(outputs.shape[1], dtype=torch.int32) * outputs.shape[0]
            len_seq = len_seq.to(config.DEVICE)
            loss = criterion(outputs, labels, len_seq, len_labels)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            torch.save(model.state_dict(),os.path.join(config.MODEL_SAVE_PATH, f'CRNN_epoch{epoch}.pth'))
            print(f'Model has been saved')
            evaluate(model=model)

    return

if __name__ == '__main__':
    train()