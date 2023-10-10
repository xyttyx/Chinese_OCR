import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import torchvision.transforms as transforms
from datasets import train_Dataset, collate_fn
from torch.utils.data import DataLoader

from models import CRNN
from utils import EncoderDecoder
import config

import cv2
from PIL import Image as _Image
from PIL import ImageTk

import numpy as np

transform = transforms.Compose([
    transforms.Resize((32,256)),  # 将图像调整为128x1024尺寸
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


model_name = 'CRNN_epoch80.pth'
model = CRNN(3000).to(config.DEVICE) #整个数据集一共约2700个不同字符，因此此处只分配3000个
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, model_name), map_location=torch.device('cpu')))
model = model.to(config.DEVICE)

# 使用tkinter库可视化
from tkinter import *
import tkinter.filedialog

img_path = ''

def produce_img(img_path):
    # 读取图像并转为灰度图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    cv2.imshow('gray',img)
    cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 双边滤波去噪
    img = cv2.bilateralFilter(img, 5, 75, 75)
    # 定义拉普拉斯算子
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # 使用卷积进行图像锐化
    img = cv2.filter2D(img, -1, kernel)
    cv2.imshow('lpls',img)
    cv2.waitKey(0)
    # img = img + dst
    # cv2.imshow('add',img)
    # cv2.waitKey(0)
    # 二值化
    cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    cv2.imshow('bi',img)
    cv2.waitKey(0)

    # 转换为PIL Image,transform期望输入PIL对象
    img = _Image.fromarray(img)

    img = transform(img)
    unloader = transforms.ToPILImage()
    show_img = unloader(img)
    show_img.show()
    img = img.unsqueeze(0) # 在最前面增加一维batch维度
    img = img.to(config.DEVICE)
    output = model(img)

    en_decoder = EncoderDecoder(config.C2I_PATH, config.I2C_PATH)
    output_str = en_decoder.TensorDecode(output[:,0,:])
    txt.insert(END,output_str+'\n')

def upload():
    img_path = tkinter.filedialog.askopenfilename()
    produce_img(img_path)    


def crop():
    H,W=500,400
    child = Toplevel(root)
    child.attributes('-topmost', 1)
    canvas = tkinter.Canvas(child, height=H, width=W)
    canvas.pack()

    img_path = tkinter.filedialog.askopenfilename()
    img = _Image.open(img_path)

    # 图像放缩
    img_w,img_h = img.size
    re_h,re_w = 0,0
    if img_h > img_w:
        re_h = H
        re_w = int(img_w * re_h / img_h)
    else:
        re_w = W
        re_h = int(img_h * re_w / img_w)
        
    img = img.resize((re_w, re_h))

    canvas.img_file = ImageTk.PhotoImage(img)    

    # 在画布上显示图像
    image = canvas.create_image(0,0,anchor='nw',image=canvas.img_file) 

    def on_button_press(event):
        global rect
        # 创建一个矩形，并保存它的ID
        rect = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red')


    def on_move_press(event):
        global rect
        # 更新矩形的坐标
        x0, y0, _, _ = canvas.coords(rect) 
        canvas.coords(rect, x0, y0, event.x, event.y)



    def on_button_release(event):
        global rect
        # 根据矩形框截取图像
        left, top, right, bottom = canvas.coords(rect)
        # print('left %d, top %d, right %d, bottom %d'%(left,top,right,bottom))
        cropped_img = img.crop((left, top, right, bottom))
        # 输出截取的图像
        # cropped_img.show()
        # 保存
        save_path = r'./demo/demo.jpg'
        cropped_img.save(save_path) 
        # 输出识别
        produce_img(save_path)  

    
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    


root= Tk()
root.title('演示')
root.geometry('800x500') # 这里的乘号不是 * ，而是小写英文字母 x
btn1 = Button(root, text='上传测试图片', command=upload, font=('黑体',10))
btn1.place(x=0,y=0,relheight=0.1,relwidth=0.15)
btn2 = Button(root, text='上传测试图片\n并框选识别', command=crop, font=('黑体',10))
btn2.place(x=0,rely=0.1,relheight=0.1,relwidth=0.15)
lb1 = tkinter.Label(root,text='测试结果如下',fg='blue',font=("黑体",30))
lb1.place(relx=0.15,y=0,relheight=0.1,relwidth=0.85)
txt=Text(root)
txt.place(relx=0.15,rely=0.1,relheight=0.85,relwidth=0.8)
txt.config(font=('楷体',18))

root.mainloop()