import torch.nn as nn
from .layer import CNNLayer, BiLSTMLayer, MobileNetV3

# 定义CRNN模型
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN,self).__init__()
        # self.cnn = CNNLayer()
        self.cnn = MobileNetV3('small')

        self.lstm_input_size = 640
        self.lstm_hidden_size = 512
        self.lstm = BiLSTMLayer(self.lstm_input_size, self.lstm_hidden_size, 2, num_classes)

        self.log_softmax = nn.LogSoftmax(dim = 2)

    def forward(self, input):
        x = self.cnn(input)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        t,n,c = x.size()
        assert c == self.lstm_input_size
        output = self.lstm(x)
        output = self.log_softmax(output)
        return output


