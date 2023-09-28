import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.EncoderDecoder import EncoderDecoder
from config import C2I_PATH, I2C_PATH

train_transform = transforms.Compose([
    transforms.Resize((32, 256)),  # 将图像调整为32x256尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class train_Dataset(Dataset):
    def __init__(self,img_folder:str,labels_folder:str):
        super(train_Dataset, self).__init__()
        self.transform = train_transform
        self.img_folder = img_folder
        self.img_names = os.listdir(img_folder)
        self.label_names = os.listdir(labels_folder)
        assert len(self.img_names) == len(self.label_names)
        for i in range(len(self.img_names)):
            assert self.img_names[i].replace('.jpg','') == self.label_names[i].replace('.txt','')
        self.labels = list()
        self.en_decoder = EncoderDecoder(C2I_PATH,I2C_PATH)
        for label_name in self.label_names:
            with open(os.path.join(labels_folder,label_name),'r',encoding='utf-8') as label_txt_file:
                tmp = label_txt_file.read()
                self.labels.append(tmp)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = Image.open(os.path.join(self.img_folder, img_name))
        img = self.transform(img)
        label = self.labels[index]
        label = self.en_decoder.StringEncode(label)
        return img, label
    
class val_Dataset(Dataset):
    def __init__(self,img_folder:str,labels_folder:str):
        super(train_Dataset, self).__init__()
        self.img_names = os.listdir(img_folder)
        self.label_names = os.listdir(labels_folder)
        assert len(self.img_names) == len(self.label_names)
        for i in range(len(self.img_names)):
            assert self.img_names[i].replace('.jpg','') == self.label_names[i].replace('.txt','')

    def __len__(self):
        return len(self.label_names)
