import torch
import torch.nn as nn
import torch.nn.functional as F
# LSTNet https://github.com/laiguokun/LSTNet
# LSTNet py3 versio https://github.com/Vsooong/pattern_recognize

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

class CNNLSTM(nn.Module):
    def __init__(self,variables,window):
        super(CNNLSTM,self).__init__()
        self.variables = variables
        self.window = window
        self.conv1 = nn.Conv1d(in_channels=window,out_channels=64,kernel_size=2).double()
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=2).double()
        self.maxpool1 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(window,128)
        self.fc1 = TimeDistributed(nn.Linear(128,64))
        self.fc2 = TimeDistributed(nn.Linear(64,1))

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = x.float()
        x = x.view(-1,1*1*1)
        # insert repeat vector here
        x = x.repeat(1,1,self.window)
        x = self.lstm1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    """model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps,features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['first_neurons'], return_sequences=True))
    model.add(TimeDistributed(Dense(params['dense_neurons'], activation='relu')))
    model.add(TimeDistributed(Dense(1)))"""

        
class LSTM(nn.Module):
    def __init__(self,hyperparams,input_dim):
        super(LSTM,self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim)
        self.lstm2 = nn.LSTM()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self,x):
        x = self.lstm1(x)
        # repeat vector
        x = self.lstm2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, variables,window):
        super(CNN, self).__init__()
        self.window = window
        self.variables = variables
        self.hw=24
        self.conv1 = nn.Conv1d(self.variables, 32, kernel_size=3).double()
        self.activate1=F.relu
        self.conv2=nn.Conv1d(32,32,kernel_size=3).double()
        self.maxpool1=nn.MaxPool1d(kernel_size=2)
        self.conv3=nn.Conv1d(32,16,kernel_size=3).double()
        self.maxpool2=nn.MaxPool1d(kernel_size=2)
        self.linear1=nn.Linear(128,100)
        self.out=nn.Linear(100,self.variables)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.output = F.sigmoid

    def forward(self, x):
        c = x.permute(0,2,1).contiguous()
        c=self.conv1(c)
        c=self.activate1(c)
        c=self.conv2(c)
        c=self.activate1(c)
        c=self.maxpool1(c)
        c=self.conv3(c)
        c=self.activate1(c)
        c=c.view(c.size(0),c.size(1)*c.size(2))
        c=self.dropout(c)
        c=self.linear1(c.float())
        c=self.dropout(c)
        out=self.out(c).float()

        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z.float())
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out