from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# LSTNet https://github.com/laiguokun/LSTNet
# LSTNet py3 versio https://github.com/Vsooong/pattern_recognize
class SpotPriceDataset(Dataset):
    def __init__(self,df,sequence_len):
        self.sequence_len = sequence_len
        self.sequences, self.labels, self.scaler = utils.prepare_data(sequence_len,df)

    def __getitem__(self,index):
        return (self.sequences[index],self.labels[index])

    def __len__(self):
        return len(self.sequences)

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=100,kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=100,out_channels=200,kernel_size=2)
        self.maxpool1 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = x.view(-1,1*1*1)
        # insert repeat vector here
        x = self.lstm1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
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


""" KERAS LSTM IMPLEMENTATION
model.add(CuDNNLSTM(params['first_neurons'],input_shape=(timesteps,features),name=f'cu_dnnlstm_0_{rand(1)[0]}'))
model.add(RepeatVector(outputs,name=f"repeat_vector_0_{rand(1)[0]}"))
model.add(CuDNNLSTM(params['second_neurons'],return_sequences=True,name=f"cu_dnnlstm_3_{rand(1)[0]}"))
model.add(TimeDistributed(Dense(params['dense_neurons'],activation='relu'),name=f"time_distributed_0_{rand(1)[0]}"))
model.add(TimeDistributed(Dense(1),name=f"time_distributed_1_{rand(1)[0]}"))
"""

""" KERAS CNN-LSTM IMPLEMENTATION
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps,features)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(outputs))
model.add(CuDNNLSTM(params['first_neurons'], return_sequences=True))
model.add(TimeDistributed(Dense(params['dense_neurons'], activation='relu')))
model.add(TimeDistributed(Dense(1)))
"""