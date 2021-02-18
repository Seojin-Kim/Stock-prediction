import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'SJ20_1': [32, 32, 'M', 32, 32, 'M'],
    'SJ20_2': [64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
}

class CNN(nn.Module):
    def __init__(self, model_code, in_channels, out_dim, act, use_bn):
        super(CNN, self).__init__()

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.TanH()
        else:
            raise ValueError("Not a valid activation function code")
        self.size = in_channels//5
        several_layers = []
        self.layers = self._make_layers(model_code, in_channels, use_bn)
        for i in range(self.size):
            several_layers.append(nn.Sequential(nn.Linear(512, out_dim)))
        self.classifier = nn.ModuleList(several_layers)
                                       #self.act,
                                       #nn.Linear(256, out_dim))
        #self.linear = nn.Sequential(nn.Linear(40, 256), self.act, nn.Linear(256, 512), self.act, nn.Linear(512, 2))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        ret = []
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
