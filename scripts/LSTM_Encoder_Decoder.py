# A LSTM Encoder-Decoder that can predict multiple timesteps from multiple timesteps. It can also be trained on as many features as we like.
# example -> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, CuDNNLSTM, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt

def prepare_data(hours_in,hours_out):
    # read dataframe from file
    combined_df = pd.read_pickle('data/combined_df_engineered.pickle')
    values = combined_df.values
    variables = values.shape[1] # 17 variables in total
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    scaler = MinMaxScaler(feature_range=(-1,1))
    
    values_nodays = values[:,:-5] # don't normalize binary values
    print(values_nodays.shape)
    norm_values = scaler.fit_transform(values_nodays)
    norm_values = np.concatenate([norm_values,values[:,-5:]],axis=1) # adding binary values back

    #test_y = norm_values[-hours_out:]
    #test_X = norm_values[-(hours_in+hours_out):len(combined_df)-hours_out]
    #train_all = norm_values[:-(hours_in+hours_out)]
    train_all = norm_values[:int(len(norm_values)*0.9)] # 90-10 train test split
    test = norm_values[-int(len(norm_values)*0.1):]
    train_X, train_y = list(),list()
    # to sequence
    in_start = 0
    for _ in range(len(train_all)):
        in_end = in_start + hours_in
        out_end = in_end + hours_out
        if out_end < len(train_all):
            train_X.append(train_all[in_start:in_end,:])
            train_y.append(train_all[in_end:out_end,0])
        in_start += 1
    return np.array(train_X), np.array(train_y), test_X, test_y, train_all, scaler

def build_model(train_X,train_y,params):
    timesteps, features, outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))
    model = Sequential()
    model.add(CuDNNLSTM(params['first_neurons'],input_shape=(timesteps,features)))
    model.add(RepeatVector(outputs))
    model.add(CuDNNLSTM(params['second_neurons'],return_sequences=True))
    model.add(TimeDistributed(Dense(params['dense_neurons'],activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer=params['optimizer'])
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

def make_prediction(model,history,hours_in):
    input_data = np.array(history)
    input_data = input_data.reshape((input_data.shape[0]*input_data.shape[1],input_data.shape[2]))
    input_X = input_data[-hours_in:,:]
    yhat = model.predict(input_X.reshape((1,input_X.shape[0],input_X.shape[1])),verbose=0)
    return yhat[0]

def model_results(model,train,test,hours_in):
    history = [x for x in train]:
    predictions = list()
    for i in range(len(test)):
        yhat_seq = make_prediction(model,history,hours_in)
        predictions.append(yhat_seq)
        history.append(test[i,:])
    predictions = np.array(predictions)

    # loopissa predictionit läpi, pitää muuttaa sitten pois skaalatuista valueista
    print(test.shape)
    for i in range(test.shape[1])


hours_in, hours_out = 24,24
train_X, train_y, test_X, test_y, train_all, scaler = prepare_data(hours_in,hours_out)
lr = 0.001
opt = Adam(lr=lr)
params = {
    'first_neurons' : 100,
    'second_neurons' : 100,
    'dense_neurons' : 100,
    'optimizer' : opt,
    'lr' : lr,
    'epochs' : 10,
    'batch_size' : 1
}
model = build_model(train_X, train_y,params)
model.save(f'models/LSTM_encoder_decoder_in{hours_in}_out{hours_out}.h5')
mae, rmse = test_model(model,test_X,test_y,scaler,hours_in,hours_out,False)

# LSTM results to csv
try:
    result_df = pd.read_csv('LSTM_test_results.csv')
except FileNotFoundError:
    result_df = pd.DataFrame(columns=['Model Name','First Neurons','Second Neurons','Dense Neurons','Optimizer','Learning Rate','Epochs','Batch Size','Hours In','Hours Out','MAE','RMSE'])
new_row = ['LSTM Encoder Decoder'] + [x for x in params.values()] + [hours_in,hours_out,mae,rmse]
result_df.loc[len(result_df)] = new_row
result_df.to_csv('LSTM_test_results.csv',index=False)
