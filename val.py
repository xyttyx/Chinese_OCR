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
import config