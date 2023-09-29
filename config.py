import os
import torch

CURRENT_EPOCH = 0
TRAIN_EPOCH = 20
BATCH_SIZE = 64

DATA_PATH = os.path.abspath('..\\Chinese_OCR_data\\datasets')
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'Train_images') 
TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'Train_label')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'Test_images') 
TEST_LABEL_PATH = os.path.join(DATA_PATH, 'Test_label') 

DICT_PATH = os.path.abspath('.\\dictionary')
I2C_DICT_NAME = 'i2c_dict.pkl'
C2I_DICT_NAME = 'c2i_dict.pkl'
I2C_PATH = os.path.join(DICT_PATH, I2C_DICT_NAME) 
C2I_PATH = os.path.join(DICT_PATH, C2I_DICT_NAME) 

MODEL_SAVE_PATH = os.path.abspath('..\\Chinese_OCR_data\\models')
TRAIN_LOG = os.path.abspath('..\\Chinese_OCR_data\\models\\train_log')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
