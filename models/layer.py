import torch
import torch.nn as nn

class CNNLayer(nn.Module):
    #使用前将Dataset的resize部分改为32先56
    def __init__(self):
        super(CNNLayer, self).__init__()
        # suppose H = 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(2,1),stride=(2,1),padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x
    

class MobileNetV3(nn.Module):
    #暂定输入图像为
    def __init__(self,mode:str):
        super(MobileNetV3, self).__init__()
        self.first_out_channels = 16
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.first_out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, out_c, se,    nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        else:
            # refer to Table 2 in paper
            mobile_setting = [
               # k, exp, out_c,  se,    nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 160,  True,  'HS', 1],
            ]
        self.main_layers = self.make_layers(mobile_setting)# B C H W 
        self.maxpool = nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
    
    def make_layers(self, mobile_setting):
        layers = []
        for i in range(len(mobile_setting)):
            if i == 0:
                layer = MobileNetV3Block(self.first_out_channels,
                                        mobile_setting[i][2],
                                        mobile_setting[i][0],
                                        mobile_setting[i][1],
                                        mobile_setting[i][3],
                                        mobile_setting[i][4],
                                        mobile_setting[i][5])
            else:
                layer = MobileNetV3Block(mobile_setting[i-1][2],
                                        mobile_setting[i][2],
                                        mobile_setting[i][0],
                                        mobile_setting[i][1],
                                        mobile_setting[i][3],
                                        mobile_setting[i][4],
                                        mobile_setting[i][5])
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self,input):
        output = self.first_conv(input)
        output = self.main_layers(output)
        output:torch.Tensor = self.maxpool(output)
        b,c,h,w = output.size()
        output = output.view(b, -1, 1, w)
        return output
    
class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, exp_size:int, use_se=True, nl:str='RE', stride:int=1,):
        super(MobileNetV3Block, self).__init__()
        self.NL = nn.ReLU6() if nl == 'RE' else nn.Hardswish()
        self.use_se = use_se
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # Depthwise convolution
        self.dwise = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            self.NL
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            self.NL
        )
        # Squeeze-and-Excitation (SE) block
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(exp_size, exp_size // 4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(exp_size // 4, exp_size, kernel_size=1, stride=1, padding=0),
                nn.Hardsigmoid()
            )
        
        # Pointwise convolution
        self.pwise = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        out = self.dwise(x)
        out = self.conv(out)
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out = self.pwise(out)
        if self.in_channels == self.out_channels and self.stride == 1:
            out += x
        return out

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_length):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out