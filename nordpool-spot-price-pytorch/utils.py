from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
def prepare_data(sequence_len,dataset):
    # read dataframe from file
    values = dataset.values
    # split data to train and test. Normalize and split data to weekly sequences

    variables = values.shape[1] # how many variables in dataset
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    #scaler = MinMaxScaler(feature_range=(0,1))
    scaler = StandardScaler()

    norm_values = scaler.fit_transform(values)

    # calculate that the week split goes even
    weeks = int(len(norm_values)/sequence_len) # or days or two days depending on sequence length
    rows_to_include = weeks*sequence_len
    norm_values = norm_values[:rows_to_include]
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))     # to weekly sequence
    train_X, train_y = list(),list()
    data = norm_values_weeklyseq.reshape((norm_values_weeklyseq.shape[0]*norm_values_weeklyseq.shape[1], norm_values_weeklyseq.shape[2]))
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + sequence_len
        out_end = in_end + sequence_len
        if out_end < len(data):
            train_X.append(data[in_start:in_end,:])
            train_y.append(data[in_end:out_end,0])
        in_start += 1
    return np.array(train_X), np.array(train_y), scaler

def mean_absolute_percentage_error(true,pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true))) * 100