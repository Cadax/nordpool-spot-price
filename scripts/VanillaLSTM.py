import generate_dataset
import pandas as pd
import pickle, math
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, RepeatVector, TimeDistributed, Flatten
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import time

def get_data():
	combined_df = generate_dataset.gen_df()
	combined_df.to_pickle("combined_df.pickle")

def load_data():
	combined_df = pd.read_pickle("combined_df.pickle")
	return combined_df

def load_test_data():
	test_df = pd.read_excel("data/elspot/elspot-prices_2019_hourly_eur.xlsx",header=2,decimal=",",dtype={'FI': np.float64})
	test_series = test_df['FI'][:-2].values
	print(len(test_series))
	test = np.array(np.split(test_series,len(test_series)/2))
	return test

def to_supervised(train,n_input,n_out = 24):
    data = train.reshape((train.shape[0]*train.shape[1],train.shape[2]))
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end < len(data):
            X.append(data[in_start:in_end,:])
            y.append(data[in_end:out_end,0])
        in_start+=1
    return np.array(X),np.array(y)

def evaluate_forecasts(actual,predicted):
    scores = list()
	# calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = math.sqrt(mse)
        scores.append(rmse)
	# calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores, actual, predicted

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

def build_model(train,test,n_input):
    train_x,train_y = to_supervised(train,n_input)
    val_x,val_y = to_supervised(test,n_input)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape((train_y.shape[0],train_y.shape[1],1))
    val_y = val_y.reshape((val_y.shape[0],val_y.shape[1],1))

    model = Sequential()
    model.add(CuDNNLSTM(200,input_shape=(n_timesteps,n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(CuDNNLSTM(200,return_sequences=True))
    model.add(TimeDistributed(Dense(100,activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mape',optimizer=Adam(lr=1e-4),metrics=['mse','mape','mae'])
    model.fit(train_x,train_y,epochs=20,shuffle=False,validation_data=(val_x,val_y),batch_size=8,verbose=1)
    return model

def forecast(model,history,n_input):
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    input_x = data[-n_input:,:]
    input_x = input_x.reshape((1,input_x.shape[0],input_x.shape[1]))
    yhat = model.predict(input_x,verbose=0)
    return yhat[0]

def evaluate_model(train, test, n_input):
	# fit model
    model = build_model(train,test, n_input)
    model.save(f'normalized_VanillaLSTM_{int(time.time())}_dataset_combined_df_stripped_swe3_shifted.pickle.h5')
    # history is a list of weekly data
    history = [x for x in train]
	# walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
        history.append(test[i, :])
	# evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores, actual, predicted = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, actual, predicted

def from_normalized(scaler,actual,predicted):
    #result_df = pd.DataFrame(columns=['Actual','Predicted'])
    #result_df['Actual'] = actual[0,:]
    #result_df['Predicted'] = predicted[0,:]
    #result_df = scaler.inverse_transform(result_df)
    with open("test_results3.txt",'w') as f:
        for i in range(actual.shape[1]):
            for j in range(actual.shape[0]):
                f.write(f"{actual[j,i]} - {predicted[j,i]} \r\n")
        

# load data
#combined_df = load_data()
#combined_df = combined_df[:-3]
combined_df = pd.read_pickle('data/shifted/combined_df_stripped_swe3_shifted.pickle')
#combined_df.drop(columns=['FI-EE','FI-NO','Rovaniemi Temperature','Jyvaskyla Temperature'],inplace=True)
print(combined_df.describe())
normalized_df = combined_df
scaler = MinMaxScaler()
scaler = scaler.fit(combined_df)
normalized_values = scaler.transform(combined_df)
normalized_df.loc[:,:] = normalized_values # normalized data
data = np.array(np.split(normalized_df.values,len(normalized_df)/24)) # training data
train = data
# test split

#test_df = normalized_df['2018-09-01':'2018-12-31'] 
#normalized_df = normalized_df.loc[(normalized_df.index < '2018-09-01')]

test_df = normalized_df[-(24*10*7):] 
normalized_df = normalized_df[:-(24*10*7)]

print(f"split train shape: {data.shape}")
test = np.array(np.split(test_df.values,len(test_df)/24))
n_input = 24
score, scores, actual, predicted = evaluate_model(train, test, n_input)
from_normalized(scaler,actual,predicted)
summarize_scores('lstm',score,scores)