import torch
import torch.nn as nn

class Conv(nn.Module):

    def __init__(self, in_chan, out_chan, kernal, dilation=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, kernal, dilation=dilation, padding=padding, groups=groups),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):

        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):]  #merge_mode = 'sum'
        return out

class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=False),
        )

    def forward(self,x):

        out,_=self.blstm(x)

        return out


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)


        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class first_layer(nn.Module):

    def __init__(self,feature_size,num_convs):
        super().__init__()
        ker = 3
        layer = []
        for i,num_conv in enumerate(num_convs):
            layer.append(nn.ConstantPad1d((2,0), 0))
            if i==0:
                
                layer.append(Conv(feature_size,num_conv,ker))
            else:
                layer.append(Conv(num_convs[i-1],num_convs[i],ker))
            
        self.first = nn.Sequential(*layer)

    def forward(self, x):
        out = self.first(x)
        return out 

class CRNN_01(nn.Module):
    
    def __init__(self,spk_num):
        super().__init__()
        feature_size = 40
        num_convs = [40,64,64,64]
        self.first_layer = first_layer(feature_size, num_convs)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=64, hidden_size=64, num_layers=3),
        )
        self.classify = nn.Linear(64,spk_num,bias=False)
    
    def forward(self,x):
        x = self.first_layer(x.permute(0,2,1))
        x = self.lstm_enc(x.permute(0,2,1))
        out = self.classify(x[:,-1])
        
        return out
    
class CRNN_02(nn.Module):
    
    def __init__(self,spk_num):
        super().__init__()
        feature_size = 40
        num_convs = [40,64,64,64]
        self.first_layer = first_layer(feature_size, num_convs)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=64, hidden_size=64, num_layers=3),
        )
        self.classify = nn.Linear(64,spk_num,bias=False)
    
    def forward(self,x):
        x = self.first_layer(x.permute(0,2,1))
        x = self.lstm_enc(x.permute(0,2,1))
        out = self.classify(x[:,-1])
        
        return out
    
class CRNN_03(nn.Module):
    
    def __init__(self,spk_num):
        super().__init__()
        feature_size = 40
        num_convs = [40,64,64,32]
        self.first_layer = first_layer(feature_size, num_convs)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=32, hidden_size=32, num_layers=3),
        )
        self.classify = nn.Linear(32,spk_num,bias=False)
    
    def forward(self,x):
        x = self.first_layer(x.permute(0,2,1))
        x = self.lstm_enc(x.permute(0,2,1))
        out = self.classify(x[:,-1])
        
        return out

class CRNN_04(nn.Module):
    
    def __init__(self,spk_num):
        super().__init__()
        feature_size = 40
        num_convs = [40,64,64,64]
        self.first_layer = first_layer(feature_size, num_convs)
        self.lstm_enc = nn.Sequential(
            Blstm(input_size=64, hidden_size=64, num_layers=3),
        )
        self.classify = nn.Linear(64,spk_num,bias=False)
    
    def forward(self,x):
        x = self.first_layer(x.permute(0,2,1))
        x = self.lstm_enc(x.permute(0,2,1))
        out = self.classify(x[:,-1])
        
        return out
    
    
