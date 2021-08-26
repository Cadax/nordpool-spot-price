from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data.dataset import Dataset

class SpotPriceDataset(Dataset):
    def __init__(self,df,sequence_len,target):
        self.sequence_len = sequence_len
        self.X = []
        self.y = []
        self.values = df.values
        self.variables = df.values.shape[1] # how many variables in dataset
        self.target = target # target column to predict AKA the spot price column
        #self.sequences, self.labels, self.scaler = self.prepare_data(sequence_len,df)

    def __getitem__(self,index):
        return (self.X[index],self.y[index])

    def __len__(self):
        return len(self.X)

    def prepare_data(self):
        # split data to train and test. Normalize and split data to weekly sequences
        labelenc = LabelEncoder()
        self.values[:,self.variables-1] = labelenc.fit_transform(self.values[:,self.variables-1])
        self.values = self.values.astype('float64')
        scaler = StandardScaler()
        norm_values = scaler.fit_transform(self.values)

        # calculate that the week split goes even
        weeks = int(len(norm_values)/self.sequence_len) # or days or two days depending on sequence length
        rows_to_include = weeks*self.sequence_len
        norm_values = norm_values[:rows_to_include]
        norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/self.sequence_len))     # to weekly sequence
        X, y = list(),list()
        data = norm_values_weeklyseq.reshape((norm_values_weeklyseq.shape[0]*norm_values_weeklyseq.shape[1], norm_values_weeklyseq.shape[2]))
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + self.sequence_len
            out_end = in_end + self.sequence_len
            if out_end < len(data):
                X.append(data[in_start:in_end,:])
                y.append(data[in_end:out_end,0])
            in_start += 1
        self.X = np.array(X)
        self.y = np.array(y)
        self.scaler = scaler

def mean_absolute_percentage_error(true,pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true))) * 100