import torch.nn as nn

class CNNLayer(nn.Module):
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
  
class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_length):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_length)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out