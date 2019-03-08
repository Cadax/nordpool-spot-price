# A LSTM Encoder-Decoder that can predict multiple timesteps from multiple timesteps. It can also be trained on as many features as we like.
# example -> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, CuDNNLSTM, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import time

def prepare_data(sequence_len,dataset):
    # read dataframe from file
    values = dataset.values
    # split data to train and test. Normalize and split data to weekly sequences

    variables = values.shape[1] # how many variables in dataset
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    scaler = MinMaxScaler(feature_range=(-1,1))
    #values_nodays = values[:,:-5] # don't normalize binary values
    #print(values_nodays.shape)
    norm_values = scaler.fit_transform(values)
    #norm_values = np.concatenate([norm_values,values[:,-5:]],axis=1) # adding binary values back

    #test_y = norm_values[-hours_out:]
    #test_X = norm_values[-(hours_in+hours_out):len(combined_df)-hours_out]
    #train = norm_values[:-(hours_in+hours_out)]

    #values = combined_df.values
    # calculate that the week split goes even
    weeks = int(len(norm_values)/sequence_len)
    rows_to_include = weeks*sequence_len
    norm_values = norm_values[:rows_to_include]
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))
    # to weekly sequence
    train = norm_values_weeklyseq[:len(norm_values_weeklyseq)-30]
    test = norm_values_weeklyseq[len(norm_values_weeklyseq)-30:]

    train_X, train_y = list(),list()
    # to sequence
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + sequence_len
        out_end = in_end + sequence_len
        if out_end < len(data):
            train_X.append(data[in_start:in_end,:])
            train_y.append(data[in_end:out_end,0])
        in_start += 1
    return np.array(train_X), np.array(train_y), test, train, scaler

def make_test(dataset,model_name,sequence_len):
    values = dataset.values
    variables = values.shape[1] 
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    scaler = MinMaxScaler(feature_range=(-1,1))
    norm_values = scaler.fit_transform(values)
    weeks = int(len(norm_values)/sequence_len)
    rows_to_include = weeks*sequence_len
    norm_values = norm_values[:rows_to_include]
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))
    # to weekly sequence
    test = norm_values_weeklyseq[:int(len(norm_values_weeklyseq)/2)]
    actual = norm_values_weeklyseq[int(len(norm_values_weeklyseq)/2):]
    #actual = test_y[:,:,0]
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
    # calculate loss
    actuals = list()
    for i in range(0,actual.shape[0]):
        test_temp = actual[i,:]
        #zeros = np.zeros((test_temp.shape[0],test.shape[2] - 1)) # train.shape[2] is number of features
        #test_temp = test_temp.reshape((test_temp.shape[0],1))
        #test_temp = np.concatenate([test_temp,zeros],axis=1)
        #print(test_temp.shape)
        act = scaler.inverse_transform(test_temp)
        actuals.append(act[:,0])
    actuals = np.array(actuals)
    test_result_df = pd.DataFrame(columns=['Actuals','Predictions'])
    actuals = actuals.reshape((actuals.shape[0]*actuals.shape[1]))
    predictions = predictions.reshape((predictions.shape[0]*predictions.shape[1]))
    test_result_df['Actuals'] = actuals
    test_result_df['Predictions'] = predictions
    return test_result_df

def build_model(train_X,train_y,params):
    timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))
    model = Sequential()
    model.add(CuDNNLSTM(params['first_neurons'],input_shape=(timesteps,features)))
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['second_neurons']))
    # two extra layers
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['second_neurons']))
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['second_neurons'],return_sequences=True))
    # two extra layers
    model.add(TimeDistributed(Dense(params['dense_neurons'],activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer=params['optimizer'])
    tensorboard = TensorBoard(log_dir=f"./logs/{int(time.time())}",histogram_freq=0,write_graph=True,write_images=False)
    model.fit(train_X, 
              train_y, 
              epochs=params['epochs'], 
              batch_size=params['batch_size'], 
              verbose=1,
              #shuffle=False,
              #validation_split=0.2,
              callbacks=[tensorboard]
              )
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
    #actual = scaler.inverse_transform(test[:,:,0])
    for i in range(0,test.shape[0]):
        test_temp = test[i,:,:]
        #test_temp = test_temp.reshape((test_temp.shape[1],test_temp.shape[2]))
        #print(f"test_temp shape {test_temp.shape}")
        act = scaler.inverse_transform(test_temp)
        #print(f"act shape {act.shape}")
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

    #week = [x for x in range(0,168)]
    #pyplot.plot(week,scores_mae, marker='o')
    #pyplot.show()

    mae = 0
    rmse = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            rmse += (actual[row,col] - predictions[row,col])**2
            mae += abs(actual[row,col] - predictions[row,col])
    rmse = np.sqrt(rmse / (actual.shape[0]*actual.shape[1]))
    mae = mae / (actual.shape[0]*actual.shape[1])
    return rmse,mae
    
if __name__ == '__main__':
    combined_df = pd.read_pickle('data/combined_df_engineered.pickle')
    combined_df.drop(columns=combined_df.columns[27:76],inplace=True) # drop onehot encoded values
    combined_df.drop(columns=['Day','FI-SE3','FI-SE1','FI-NO','Consumption','yhat_lower','trend','^N100 Close','^OMXSPI Close','FORTUM Close','^OMXH25 Close','Hydro Reservoir FI','Wind Power Production FI','Rovaniemi Temperature','Jyvaskyla Temperature','Helsinki Temperature'],inplace=True) #drop features with low importance score
    print(combined_df.columns)
    sequence_len = 24*3 # sequence length, week is 24 hours * 7 
    #hours_in, hours_out = 24,24
    train_X, train_y, test, train, scaler = prepare_data(sequence_len,combined_df)
    #lr = 0.001
    #opt = Adam(lr=lr)

    params_list = list()
    params_list.append({
        'first_neurons' : 30,
        'second_neurons' : 30,
        'dense_neurons' : 15,
        'lr' : 0.001,
        'optimizer' : Adam(lr=0.001),
        'epochs' : 30,
        'batch_size' : 1
    })
    params_list.append({
        'first_neurons' : 60,
        'second_neurons' : 60,
        'dense_neurons' : 30,
        'lr' : 0.001,
        'optimizer' : Adam(lr=0.001),
        'epochs' : 30,
        'batch_size' : 8
    })
    params_list.append({
        'first_neurons' : 500,
        'second_neurons' : 500,
        'dense_neurons' : 250,
        'lr' : 0.0001,
        'optimizer' : Adam(lr=0.0001),
        'epochs' : 50,
        'batch_size' : 16
    })

    for params in params_list:
        model = build_model(train_X, train_y,params)
        model_path = f"models/LSTM_encoder_decoder_sequence{sequence_len}-{int(time.time())}.h5"
        model.save(model_path)
        model_name = model_path.split('/')[1]
        #model = load_model('models/LSTM_encoder_decoder_sequence168.h5')
        #mae, rmse = test_model(model,test,scaler,sequence,False)
        rmse, mae = model_results(model,train,test,sequence_len,scaler)
        print(f"RMSE: {rmse}, MAE: {mae}")

        # LSTM results to csv
        try:
            result_df = pd.read_csv('LSTM_test_results.csv')
        except FileNotFoundError:
            result_df = pd.DataFrame(columns=['Model Name','First Neurons','Second Neurons','Dense Neurons','Optimizer','Learning Rate','Epochs','Batch Size','Sequence Length','MAE','RMSE'])
        new_row = [model_name] + [x for x in params.values()] + [sequence_len,mae,rmse]
        result_df.loc[len(result_df)] = new_row
        result_df.to_csv('LSTM_test_results.csv',index=False)
