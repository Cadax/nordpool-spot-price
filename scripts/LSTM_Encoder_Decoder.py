# A LSTM Encoder-Decoder that can predict multiple timesteps from multiple timesteps. It can also be trained on as many features as we like.
# example -> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, CuDNNLSTM, RepeatVector, TimeDistributed, Dropout, BatchNormalization, Bidirectional, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Permute, Flatten, Input, PReLU, Concatenate, Activation, Masking, Reshape, GRU
from tensorflow.python.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import L1L2
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.python.keras.utils.vis_utils import plot_model
from decimal import Decimal as D
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import time
from numpy.random import rand
import argparse
import tensorflow

def mean_absolute_percentage_error(true,pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true))) * 100

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

def make_test(dataset,model_name,sequence_len):
    dates = dataset.index #remember datetimes from index
    values = dataset.values
    variables = values.shape[1] 
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    #scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    norm_values = scaler.fit_transform(values)
    weeks = int(len(norm_values)/sequence_len)
    rows_to_include = weeks*sequence_len
    norm_values = norm_values[:rows_to_include]
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))
    test = norm_values_weeklyseq
    actual = norm_values_weeklyseq
    model = load_model(f'../scripts/models/{model_name}')

    history = [x for x in test]
    predictions = list()
    for i in range(len(actual)):
        yhat_seq = make_prediction(model,history,sequence_len)
        zeros = np.zeros((yhat_seq.shape[0],test.shape[2] - 1)) # train.shape[2] is number of features
        real_prediction = np.concatenate([yhat_seq,zeros],axis=1)
        real_prediction = scaler.inverse_transform(real_prediction)
        predictions.append(real_prediction[:,0])
        history.append(actual[i,:])

    predictions = np.array(predictions)
    actuals = list()
    for i in range(0,actual.shape[0]):
        test_temp = actual[i,:]
        act = scaler.inverse_transform(test_temp)
        actuals.append(act[:,0])
    actuals = np.array(actuals)
    test_result_df = pd.DataFrame(columns=['Actuals','Predictions'])

    actuals = actuals.reshape((actuals.shape[0]*actuals.shape[1]))
    predictions = predictions.reshape((predictions.shape[0]*predictions.shape[1]))
    test_result_df['Actuals'] = actuals
    test_result_df['Predictions'] = predictions
    if len(actuals) < len(dates):
        dates = dates[:len(actuals)]
    test_result_df.set_index(dates,inplace=True)
    test_result_df.to_csv(f'../scripts/test_results/{model_name}_results.csv',index=True,index_label='Date')
    return test_result_df

def build_model(train_X,train_y,params):
    reg = L1L2(l1=0.01,l2=0.01)
    train_X, val_X = train_X[:-(24*10*7)], train_X[-(24*10*7):] # validation set same as test dataset that is defined later
    train_y, val_y = train_y[:-(24*10*7)], train_y[-(24*10*7):]
    timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))
    val_y = val_y.reshape((val_y.shape[0],val_y.shape[1],1))

    model = Sequential()
    model.add(CuDNNLSTM(params['first_neurons'],input_shape=(timesteps,features),name=f'cu_dnnlstm_0_{rand(1)[0]}'))
    model.add(RepeatVector(outputs,name=f"repeat_vector_0_{rand(1)[0]}"))
    # two extra layers
    model.add(CuDNNLSTM(params['first_neurons'],name=f"cu_dnnlstm_1_{rand(1)[0]}"))
    model.add(RepeatVector(outputs,name=f"repeat_vector_1_{rand(1)[0]}"))
    model.add(CuDNNLSTM(params['first_neurons'],name=f"cu_dnnlstm_2_{rand(1)[0]}"))
    model.add(RepeatVector(outputs,name=f"repeat_vector_2_{rand(1)[0]}"))
    # two extra layers
    model.add(CuDNNLSTM(params['second_neurons'],return_sequences=True,name=f"cu_dnnlstm_3_{rand(1)[0]}"))
    model.add(TimeDistributed(Dense(params['dense_neurons'],activation='relu'),name=f"time_distributed_0_{rand(1)[0]}"))
    model.add(TimeDistributed(Dense(1),name=f"time_distributed_1_{rand(1)[0]}"))
    
    model.compile(loss='mae', optimizer=params['optimizer'], metrics=['mse','mape','mae'])
    
    #tensorboard = TensorBoard(log_dir=f"./logs/{int(time.time())}",histogram_freq=0,write_graph=True,write_images=False)
    early_stopping = EarlyStopping(monitor='val_mean_absolute_error',mode='min',patience=5)
    checkpoint_name = f'{int(time.time())}.h5'
    model_checkpoint = ModelCheckpoint(f'./checkpoints/{checkpoint_name}',monitor='val_mean_absolute_error',mode='min',save_best_only=True,verbose=1)
    print(model.summary())
    model.fit(train_X, 
              train_y, 
              epochs=params['epochs'], 
              batch_size=params['batch_size'], 
              verbose=1,
              shuffle=False,
              validation_data=(val_X,val_y),
              callbacks=[early_stopping,model_checkpoint]
              )
    #plot_model(model, to_file='../images/LSTM_network.png', show_shapes=True, show_layer_names=False)
    return model, checkpoint_name

def build_model_cnnlstm(train_X,train_y,params):
    reg = L1L2(l1=0.01,l2=0.01)
    train_X, val_X = train_X[:-(24*10*7)], train_X[-(24*10*7):] # validation set same as test dataset that is defined later
    train_y, val_y = train_y[:-(24*10*7)], train_y[-(24*10*7):]
    timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))
    val_y = val_y.reshape((val_y.shape[0],val_y.shape[1],1))


    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps,features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['first_neurons'], return_sequences=True))
    model.add(TimeDistributed(Dense(params['dense_neurons'], activation='relu')))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mae', optimizer=params['optimizer'], metrics=['mape','mae'])
    #tensorboard = TensorBoard(log_dir=f"./logs/{int(time.time())}",histogram_freq=0,write_graph=True,write_images=False)
    early_stopping = EarlyStopping(monitor='val_mean_absolute_error',mode='min',patience=10)
    checkpoint_name = f'{int(time.time())}.h5'
    model_checkpoint = ModelCheckpoint(f'./checkpoints/{checkpoint_name}',monitor='val_mean_absolute_error',mode='min',save_best_only=True,verbose=1)
    model.fit(train_X, 
              train_y, 
              epochs=params['epochs'], 
              batch_size=params['batch_size'], 
              verbose=1,
              shuffle=False,
              validation_data=(val_X,val_y),
              callbacks=[early_stopping,model_checkpoint]
              )
    #plot_model(model, to_file='../images/CNNLSTM_network.png', show_shapes=True, show_layer_names=False)
    return model, checkpoint_name

def build_model_convlstm(train_X,train_y,params):
    length = 24
    timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))

    # reshape into subsequences [samples, time steps, rows, cols, channels]
    train_X = train_X.reshape((train_X.shape[0], timesteps, 1, length, features))
	# define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(timesteps, 1, length, features)))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(CuDNNLSTM(params['first_neurons'], activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(params['dense_neurons'], activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_X, train_y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)
    return model

def test_model(model,test_X,test_y,scaler,hours_in,hours_out,test_2019):
    if test_2019:
        # test from 2019 values, outside of dataset
        test_df = pd.read_pickle('data/test_df.pickle') # load test_data
        norm_inputvalues = scaler.transform(combined_df[-hours_in:].values)
        norm_forecast = model.predict(norm_inputvalues.reshape((1,norm_inputvalues.shape[0],norm_inputvalues.shape[1])))
        # pad forecast with zeroes and inverse transform
        zeros = np.zeros((hours_out,17))
        for i in range(0,norm_forecast.shape[1]):
            concat = np.concatenate([norm_forecast[0,i,:],zeros[i,:][1:]])
            norm_df.loc[i] = concat
        forecast = scaler.inverse_transform(norm_df.values)
        forecast = forecast[1:,0] # from 31.12.2018 22:00, ignore first value to get one week of january
        # actual values
        inv_actual = test_df['FI'][:hours_out-1].values # january 2019 values from pickle
        inv_actual = inv_actual.reshape(-1,1)
        inv_actual = np.concatenate([inv_actual,zeros],axis=1)
        actual = scaler.inverse_transform(inv_actual)
        actual = actual[:,0]
        plt.figure(figsize=(14,6))
        plt.plot(forecast)
        plt.plot(actual)
        plt.legend(['Forecast','Actual'])
        plt.show()
        print(f"Test MAE: {mean_absolute_error(actual,forecast)}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(actual,forecast))}")
    else:
        # testing from combined_df
        test_X_nodays = test_X[:,:-5]
        norm_inputvalues = scaler.transform(test_X_nodays)
        norm_inputvalues = np.concatenate([norm_inputvalues,test_X[:,-5:]],axis=1)
        scaled_forecast = model.predict(norm_inputvalues.reshape((1,norm_inputvalues.shape[0],norm_inputvalues.shape[1])))
        scaled_forecast = scaled_forecast.reshape((scaled_forecast.shape[1],scaled_forecast.shape[0]))
        # pad forecast with zeroes and inverse transform
        zeros = np.zeros((hours_out,test_X.shape[1] - 6)) # substract the binary columns
        print(zeros.shape)
        print(scaled_forecast.shape)
        scaled_forecast = np.concatenate([scaled_forecast,zeros],axis=1)
        forecast = scaler.inverse_transform(scaled_forecast)
        forecast = forecast[:,0]

        scaled_actual = test_y[:,:-5] # from combined_df, convert back from scaled
        actual = scaler.inverse_transform(scaled_actual)
        actual = actual[:,0]
        mae = mean_absolute_error(actual,forecast)
        rmse = np.sqrt(mean_squared_error(actual,forecast))
        print(f"Test MAE: {mae}")
        print(f"Test RMSE: {rmse}")
        
        plt.figure(figsize=(14,6))
        plt.plot(forecast)
        plt.plot(actual)
        plt.legend(['Forecast','Actual'])
        plt.show()
        return mae, rmse

def make_prediction(model,history,sequence_len):
    input_data = np.array(history)
    input_data = input_data.reshape((input_data.shape[0]*input_data.shape[1],input_data.shape[2]))
    input_X = input_data[-sequence_len:,:]
    yhat = model.predict(input_X.reshape((1,input_X.shape[0],input_X.shape[1])),verbose=0)
    return yhat[0]

def model_results(model,train,test,sequence_len,scaler):
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        yhat_seq = make_prediction(model,history,sequence_len)
        zeros = np.zeros((yhat_seq.shape[0],train.shape[2] - 1)) # train.shape[2] is number of features
        real_prediction = np.concatenate([yhat_seq,zeros],axis=1)
        real_prediction = scaler.inverse_transform(real_prediction)
        predictions.append(real_prediction[:,0])
        history.append(test[i,:])
    predictions = np.array(predictions)
    # calculate loss
    actual = list()
    for i in range(0,test.shape[0]):
        test_temp = test[i,:,:]
        act = scaler.inverse_transform(test_temp)
        actual.append(act[:,0])
    actual = np.array(actual)
    scores_mae = list()
    scores_rmse = list()
    for i in range(actual.shape[1]):
        rmse = mean_squared_error(actual[:,i],predictions[:,i])
        mae = mean_absolute_error(actual[:,i],predictions[:,i])
        rmse = np.sqrt(rmse)
        scores_rmse.append(rmse)
        scores_mae.append(mae)

    mae = 0
    rmse = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            rmse += (actual[row,col] - predictions[row,col])**2
            mae += abs(actual[row,col] - predictions[row,col])
    rmse = np.sqrt(rmse / (actual.shape[0]*actual.shape[1]))
    mae = mae / (actual.shape[0]*actual.shape[1])
    return rmse,mae
    


def main(args):
    params_list = list()
    if args.hparams is None:
        params_list.append({
        'first_neurons' : 1000,
        'second_neurons' : 1000,
        'dense_neurons' : 1000,
        'lr' : 1e-5,
        'optimizer' : 'Adam',
        'epochs' : 5,
        'batch_size' : 168,
        'sequence_len' : 168
        })
    else:
        hparams_dict = {}
        with open('../scripts/hyperparameters.txt') as hparams:
            for line in hparams:
                split = line.split(':')
                hparam = split[0].strip()
                if hparam == 'lr':
                    hparams_dict[hparam] = D(split[1].strip())
                elif hparam == 'optimizer':
                    hparams_dict[hparam] = str(split[1].strip())
                else:
                    hparams_dict[hparam] = int(split[1].strip())
        params_list.append(hparams_dict)
        
    for params in params_list:
        # initialize optimizer from params
        params['optimizer'] = tensorflow.keras.optimizers.get(params['optimizer'].lower())
        K.set_value(params['optimizer'].lr, params['lr'])
        combined_df = pd.read_pickle(args.dataset)
        sequence_len = int(params['sequence_len'])
        columns_to_drop = [x for x in combined_df.columns if 'Spot' and 'T-' in x and int(x.split('-')[1]) < sequence_len]
        combined_df.drop(columns=columns_to_drop,inplace=True)
        print(f"Features in dataset: {len(combined_df.columns)-1}")
        train_X,train_y,scaler = prepare_data(sequence_len,combined_df)

        if args.cnn is True:
            model,checkpoint_name = build_model_cnnlstm(train_X,train_y,params)
        else:
            model,checkpoint_name = build_model(train_X, train_y,params)

        dataset_name = args.dataset.split('/')[-1]
        if args.cnn is True:
            model_path = f"models/CNNLSTM_encdec_stdscaler_lossMAE_dataset{dataset_name}_sequence{sequence_len}-{int(time.time())}.h5"
        else:
            model_path = f"models/LSTM_encdec_stdscaler_lossMAE_dataset{dataset_name}_sequence{sequence_len}-{int(time.time())}.h5"
        model.save(model_path)
        model_name = model_path.split('/')[1]
        
        # print error results from test data
        test_df = combined_df[-(24*10*7):]
        columns_to_drop = [x for x in test_df.columns if 'Spot' and 'T-' in x and int(x.split('-')[1]) < sequence_len]
        test_df.drop(columns=columns_to_drop,inplace=True)
        test_result_df = make_test(test_df,model_name,sequence_len)
        mae = mean_absolute_error(test_result_df['Actuals'],test_result_df['Predictions'])
        rmse = np.sqrt(mean_squared_error(test_result_df['Actuals'],test_result_df['Predictions']))
        mape = mean_absolute_percentage_error(test_result_df['Actuals'],test_result_df['Predictions'])
        
        print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")
        # Results to csv
        try:
            result_df = pd.read_csv('LSTM_test_results2.csv')
        except FileNotFoundError:
            result_df = pd.DataFrame(columns=['Model Name','Checkpoint Name','First Neurons','Second Neurons','Dense Neurons','Optimizer','Learning Rate','Epochs','Batch Size','Sequence Length','MAE','RMSE','MAPE'])
        new_row = [model_name] + [checkpoint_name] + [x for x in params.values()] + [mae,rmse,mape]
        result_df.loc[len(result_df)] = new_row
        result_df.to_csv('LSTM_test_results2.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,help='Name of the model')
    parser.add_argument('--dataset',type=str,help='Name of the dataset to train with')
    parser.add_argument('--cnn',help='Use the CNN-LSTM model to train',action='store_true')
    parser.add_argument('--hparams',type=str,help='Load hyperparameters from file')
    args = parser.parse_args()
    
    main(args)