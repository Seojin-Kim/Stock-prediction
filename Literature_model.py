import torch.nn as nn
import torch.nn.functional as F
import torch


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'SJ20_1': [64, 64, 'M', 128, 128, 'M'],
    'SJ20_2': [64, 64, 'M'],
}

class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, dropout, use_bn):
        super(LSTM, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        #self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn 
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True)
        self.hidden = self.init_hidden()
        #self.regressor = self.make_regressor()
        #self.linear = nn.Linear(160, 2)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    

    
    def forward(self, x):
        x = x.transpose(1,2)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out[:, -1, :].view(self.batch_size, -1)



class CNN(nn.Module):
    def __init__(self, model_code, in_channels, act, use_bn):
        super(CNN, self).__init__()

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.TanH()
        else:
            raise ValueError("Not a valid activation function code")

        self.layers = self._make_layers(model_code, in_channels, use_bn)
#         self.classifer = nn.Sequential(nn.Linear(512, 256),
#                                       self.act,
#                                       nn.Linear(256, out_dim))
        #self.linear = nn.Sequential(nn.Linear(40, 256), self.act, nn.Linear(256, 512), self.act, nn.Linear(512, 2))

    def forward(self, x):
        x = self.layers(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifer(x)
        return x

    def _make_layers(self, model_code, in_channels, use_bn):
        layers = []
        for x in cfg[model_code]:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels=in_channels,
                                     out_channels=x,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)]
                if use_bn:
                    layers += [nn.BatchNorm1d(x)]
                layers += [self.act]
                in_channels = x
            #print(x)
        return nn.Sequential(*layers)




class SJ_net(nn.Module):
    def __init__(self, model_code1, model_code2, in_channels, act, use_bn, input_dim, hidden_dim, output_dim, num_lstm_layers, batch_size, dropout, size):
        super(SJ_net, self).__init__()
        self.CNN1 = CNN(model_code1, in_channels, act, use_bn)
        self.CNN2 = CNN(model_code2, in_channels, act, use_bn)
        self.LSTM1 = LSTM(cfg[model_code1][-2], hidden_dim, num_lstm_layers, batch_size, dropout, use_bn)
        self.LSTM2 = LSTM(cfg[model_code2][-2], hidden_dim, num_lstm_layers, batch_size, dropout, use_bn)
        self.LSTM3 = LSTM(input_dim, hidden_dim, num_lstm_layers, batch_size, dropout, use_bn)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.TanH()
        else:
            raise ValueError("Not a valid activation function code")
        self.size = size
        several_layers = []

        for i in range(self.size):
            several_layers.append(nn.Linear(3*hidden_dim, output_dim))
        self.classifier = nn.ModuleList(several_layers)


    def forward(self, x):
        y = self.CNN1(x)
        y = self.LSTM1(y)
        z = self.CNN2(x)
        z = self.LSTM2(z)
        w = self.LSTM3(x)
        x = torch.cat((y, z, w), 1)
        for i in range(self.size):
            if i==0:
                y = self.classifier[i](x)
                ret = F.log_softmax(y, dim=1)
                ret = ret.unsqueeze(2)
            else:
                y = self.classifier[i](x)
                tmp_ret = F.log_softmax(y, dim=1)
                ret = torch.cat((ret,tmp_ret.unsqueeze(2)),axis=2)
        return ret
