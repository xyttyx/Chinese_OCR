# Chinese_OCR
手写中文文本行识别，使用CRNN+CTC，在HWBD2.x数据集上accuracy达到**0.954**

## 详解CRNN+CTC
### CRNN部分
CRNN采用CNN网络加RNN网络进行联合识别。首先使用CNN对网络的特征进行提取，其次使用RNN在时间序列上（此处的时间序列即为图片的横向坐标）对每个感受野下提取的特征进行识别。
（未完待续）

### CTC函数
虽然在PyTorch之中，已经集成了CTCLoss这一损失函数，但了解原理对于我们的学习至关重要。下面将讲解CTCLoss的原理。
（未完待续）
***
## 训练方式
使用requirements.txt安装必要的包，在命令行中输入
`pip install -r requirements.txt`

请在当前工程文件夹的上一级文件夹中，创建数据集文件夹，默认文件夹名称为**Chinese_OCR_data**。此文件夹默认结构如下  
```
Chinese_OCR_data  
  |--datasets  
  |    |--Test_images  
  |    |--Test_label  
  |    |--Train_images  
  |    |--Train_label  
  |--model   
```

请确认图片文件夹和标签文件夹下图片与标签名称相同，内容一一对应    
请注意，默认标签为.txt文件，图片为.jpg文件。如果使用HWBD2.x数据集，则可以使用utils下的trans_dgrl.py将.dgrl文件分离为图片和标签。
**如需使用trans_dgrl.py，请确认并修改文件中指向的路径**

在命令行中输入
`python train.py`
开始训练

## 测试方式
使用evaluate.py，会使用测试集中的数据进行评估。评估所输出的accuracy
计算公式如下   

***accuracy = average(1 - 编辑距离/标签字符串长度)***  

编辑距离的定义和计算方式可自行搜索，实现方式放在util.editDistance之中。
***
#### 闲言碎语
这个网络与原始CRNN的区别在于：

1. 使用了9层卷积代替7层卷积；
2. 使用了双层的biLSTM作为转录层。

其中，9层卷积神经网络的设计参考了论文《Attention机制在脱机中文手写体文本行识别中的应用》。这篇论文在单层biLSTM后加入了Attention机制的seq2seq，其acc为0.9576，而本模型在训练80轮后效果最好的模型acc为0.954。