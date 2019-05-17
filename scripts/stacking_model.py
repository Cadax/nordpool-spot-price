# https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
import fbprophet, argparse, xgboost, os, LSTM_Encoder_Decoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pyramid.arima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras import backend as K
import numpy as np

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res / (SS_tot + K.epsilon()))

def ar_forecast(history,forecast_horizon):
    history = np.array(history)
    model = AR(history).fit()
    yhat = model.predict(len(history),len(history)+(forecast_horizon-1))
    return list(yhat)

def holt_winters_forecast(history,sequence_len):
    history = np.array(history)
    model = ExponentialSmoothing(history, seasonal='add', seasonal_periods=sequence_len).fit()
    yhat = model.predict(len(history),len(history)+sequence_len-1)
    return list(yhat)

def xgboost_forecast(history,sequence_len):
    history = np.array(history)
    model = xgboost.XGBRegressor(objective='reg:linear',colsample_bytree=0.7,learning_rate=0.1,max_depth=2,alpha=10,n_estimators=50,subsample=0.9)
    model.fit(history).fit()
    yhat = model.predict(len(history),len(history)+sequence_len-1)
    return list(yhat)

def ar_prediction(training_data,test,forecast_horizon):
    #combined_df = pd.read_pickle('combined_df_engineered_T-24.pickle')
    #test_data = pd.read_pickle('test_df.pickle') # January 2019 test data
    #train = pd.DataFrame(data=combined_df['Spot']['2018'].values,index=combined_df['2018'].index)
    training_data = training_data['Spot']
    test = test['Spot']
    #test = pd.DataFrame(data=test_data['FI'].values,index=test_data.index)
    # training data 2018
    predictions_ar = []
    history_ar = [spot_price for spot_price in training_data]
    hour_counter = forecast_horizon
    for i in range(int(len(test)/forecast_horizon)):
        yhat_list = ar_forecast(history_ar,forecast_horizon)
        for yhat in yhat_list:
            predictions_ar.append(yhat)
        for x in range(hour_counter-forecast_horizon,hour_counter):
            history_ar.append(test.iloc[x])
        hour_counter += forecast_horizon
    return np.array(predictions_ar)

def prophet_prediction(train,test):
    train = train['Spot']
    test = test['Spot']
    train = train['2018']
    spot = pd.DataFrame(columns=['ds','y'])
    spot['y'] = train
    spot['ds'] = train.index
    spot.index = pd.RangeIndex(len(spot.index))
    scale = 0.001
    prophet = fbprophet.Prophet(changepoint_prior_scale=scale)
    prophet.fit(spot)
    forecast = prophet.make_future_dataframe(periods=len(test),freq='H')
    forecast = prophet.predict(forecast)
    forecast.set_index(['ds'],inplace=True)
    forecast = forecast[-(24*10*7):]
    return forecast

def xgboost_prediction(train,test,sequence_len):
    train = train
    test = test['Spot']
    #test = pd.DataFrame(data=test_data['FI'].values,index=test_data.index)
    # training data 2018
    predictions_xg = list()
    history_xg = [x for x in train]
    hour_counter = sequence_len - 1 # 24-1 to prevent index out of bounds
    for i in tqdm(range(int(len(test)/sequence_len))):
        print(history_xg)
        yhat_list = xgboost_forecast(history_xg,sequence_len)
        for yhat in yhat_list:
            predictions_xg.append(yhat)
        for x in range(hour_counter-sequence_len,hour_counter):
            history_xg.append(test.iloc[x])
        #history.append(test[hour_counter] for hour_counter in range(hour_counter,hour_counter+24))
        hour_counter += sequence_len # 24 hour timestep
    return np.array(predictions_ar)

def hw_prediction(train,test,sequence_len):
    predictions_hw = list()
    train = train['2018']
    history_hw = [x for x in train.iloc[:,0]]
    hour_counter = sequence_len - 1 # 24-1 to prevent index out of bounds
    for i in tqdm(range(int(len(test)/sequence_len))):
        yhat_list = holt_winters_forecast(history_hw,sequence_len)
        for yhat in yhat_list:
            predictions_hw.append(yhat)
        for x in range(hour_counter-sequence_len,hour_counter):
            history_hw.append(test.iloc[x,0])
        #history.append(test[hour_counter] for hour_counter in range(hour_counter,hour_counter+24))
        hour_counter += sequence_len # 24 hour timestep
    return np.array(predictions_hw)

def sarimax_prediction(train,test):
    df = train['2018-06':]
    df = df.asfreq(freq='H')
    test_df = test.asfreq(freq='H')
    df.drop(columns=df.columns[1:],inplace=True)
    sarima = SARIMAX(df,order=(7,1,7),seasonal_order=(7,1,7,24),enforce_stationarity=False,enforce_invertibility=False,freq='H').fit()
    pred = sarima.predict(test.index[0],test.index[-1])
    return pred

def autoarima_prediction(train,test):
    train = train['2018']
    train = train['Spot']
    model = auto_arima(train,start_p=1,start_q=1,start_P=1,start_Q=1,max_q=2,max_p=2,m=12,d=1,D=1,stationary=False,seasonal=True,n_jobs=-1)
    print(model.aic())
    model.fit(train)
    forecast = model.predict(n_periods=len(test))
    return forecast


# create a dataset out of other model's predictions
def create_dataset(model_names):
    stack = None
    for model_name in model_names:
        dataset_name = model_name.split('dataset')[1].split('pickle')[0] + "pickle"
        model = load_model(f"models/{model_name}")
        dataset = pd.read_pickle(f'data/{dataset_name}')
        dataset = dataset[-(24*10*7):]
        predictions_lstm = make_prediction(dataset,model,24)
        print(predictions.shape)
        if stack is None:
            stack = predictions
        else:
            stack = np.dstack((stack,predictions))
        print(stack.shape)
    stack = stack.reshape((stack.shape[0], stack.shape[1]*stack_shape[2]))
    return stack


def lstm_prediction(test_dataset,model_name,sequence_len):
    #dates = dataset.index #remember datetimes from index
    values = test_dataset.values
    variables = values.shape[1]
    labelenc = LabelEncoder()
    values[:,variables-1] = labelenc.fit_transform(values[:,variables-1])
    values = values.astype('float64')
    model = load_model("models/" + model_name)
    if "stdscaler" in model_name:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(values)
    weeks = int(len(norm_values)/sequence_len)
    rows_to_include = weeks*sequence_len
    norm_values = norm_values[:rows_to_include]
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))
    # to weekly sequence
    test = norm_values_weeklyseq
    actual = norm_values_weeklyseq

    history = [x for x in test]
    predictions = list()
    for i in range(len(actual)):
        yhat_seq = LSTM_Encoder_Decoder.make_prediction(model,history,sequence_len)
        zeros = np.zeros((yhat_seq.shape[0],test.shape[2] - 1)) # train.shape[2] is number of features
        real_prediction = np.concatenate([yhat_seq,zeros],axis=1)
        real_prediction = scaler.inverse_transform(real_prediction)
        predictions.append(real_prediction[:,0])
        history.append(actual[i,:])
    predictions = np.array(predictions)
    predictions = predictions.reshape((predictions.shape[0]*predictions.shape[1]))
    del model
    return predictions

def mean_of_predictions(test,predictions,filename):
    preds = np.array(predictions)
    mean = preds.mean(axis=1)
    result_df = pd.DataFrame(index=test.index)
    result_df['Actuals'] = test['Spot']
    result_df['Predictions'] = mean
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')

def median_of_predictions(test,predictions,filename):
    preds = np.array(predictions)
    median = np.median(preds,axis=1)
    result_df = pd.DataFrame(index=test.index)
    result_df['Actuals'] = test['Spot']
    result_df['Predictions'] = median
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')

def grid_search_xgboost(X,y):
    params = {
        'objective' : ['reg:linear'],
        'n_estimators' : np.linspace(5,200,40,dtype=int),
        'learning_rate' : [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'min_child_weight': [1, 3, 5, 7, 9],
        'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'silent': [1],
        'max_depth': [2, 3, 4, 5, 6, 7]
    }
    """
    fit_params = {
        "eval_metric": "mae",
        "eval_set" : [[test_X,test_y['Spot'].values]]
    }
    """

    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    xgbregressor = xgboost.XGBRegressor()
    grid_search = RandomizedSearchCV(xgbregressor,params,scoring='r2',n_jobs=-1,verbose=1,cv=2,n_iter=200000)
    print(X.values.shape)
    print(y.shape)
    y = y.values.reshape((y.shape[0],1))
    print(y.shape)
    grid_search.fit(X.values,y,verbose=False)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

def xgboost_metalearner(test,predictions,filename):
    train_X, train_y = predictions[:int(len(predictions)/2)], test[:int(len(test)/2)]
    test_X, test_y = predictions[-int(len(predictions)/2):], test[-int(len(test)/2):]
    
    test = test[-int(len(test)/2):]
    xgbregressor = xgboost.XGBRegressor(objective='reg:linear',
                                       min_child_weight=9,
                                       colsample_bytree=0.7,
                                       learning_rate=0.61,
                                       max_depth=2,
                                       subsample=0.5,
                                       n_estimators=5
                                       )
    xgbregressor.fit(train_X,train_y['Spot'])
    result_df = pd.DataFrame(index=test.index)
    result_df['Actuals'] = test_y['Spot']
    result_df['Predictions'] = xgbregressor.predict(test_X)
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')

def grid_search_random_forest(train_X,train_y):
    params = { 
    'n_estimators': np.linspace(100,1000,11,dtype=int),
    'min_samples_split': np.linspace(2,16,9,dtype=int),
    'min_samples_leaf' : np.linspace(1,10,10,dtype=int),
    'max_features' : ['auto','sqrt'],
    'max_depth' : np.linspace(10,100,11,dtype=int),
    'bootstrap' : [True,False],
    'n_jobs' : [-1]
    }
    forest = RandomForestRegressor()
    grid_search = RandomizedSearchCV(forest,params,scoring='r2',n_jobs=-1,verbose=1,cv=2,n_iter=1000)
    grid_search.fit(train_X,train_y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)


def random_forest_metalearner(test,predictions,filename):
    train_X, train_y = predictions[:int(len(predictions)/2)], test[:int(len(test)/2)]
    test_X, test_y = predictions[-int(len(predictions)/2):], test[-int(len(test)/2):]
    #grid_search_random_forest(train_X,train_y)

    forest = RandomForestRegressor(n_estimators=460,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_features='sqrt',
                                   max_depth=19,
                                   bootstrap=True,
                                   random_state=42,
                                   n_jobs=-1)
    forest.fit(train_X, train_y['Spot'])
    result_df = pd.DataFrame(index=test.index[:len(test_y)])
    result_df['Actuals'] = test_y['Spot'].values
    result_df['Predictions'] = forest.predict(test_X)
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')


def neural_network_metalearner_wo_scaling(test,preds,filename):
    test = test['Spot']
    #preds = np.array(predictions)

    train_X, train_y = preds[:int(len(preds)/2)], test[:int(len(test)/2)]
    test_X, test_y = preds[-int(len(preds)/2):], test[-int(len(test)/2):]
    test = test[int(len(test)/2):]
    features = train_X.shape[1]

    model = Sequential()
    model.add(Dense(100,input_shape=(features,),activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(features,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer=Adam(lr=1e-5),metrics=['mape'])
    model.fit(train_X,train_y,batch_size=1,epochs=30,validation_data=(test_X,test_y))

    result_df = pd.DataFrame(index=test.index)
    result_df['Actuals'] = test
    result_df['Predictions'] = model.predict(test_X)
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')



def neural_network_metalearner(test,preds,filename):
    test = test[:len(preds)]
    test = test['Spot']
    test_index = test.index
    #preds_scaler = MinMaxScaler(feature_range=(0,1))
    #test_scaler = MinMaxScaler(feature_range=(0,1))
    preds_scaler = StandardScaler()
    test_scaler = StandardScaler()
    preds = preds_scaler.fit_transform(preds)
    test = test_scaler.fit_transform(test.values.reshape(-1,1))
    #preds = preds.reshape(-1,1)
    #test = test.reshape(-1,1)
    print(preds.shape)
    print(test.shape)
    train_X, train_y = preds[:int(len(preds)/2)], test[:int(len(test)/2)]
    test_X, test_y = preds[-int(len(preds)/2):], test[-int(len(test)/2):]
    test = test[int(len(test)/2):]
    features = train_X.shape[1]

    model = Sequential()
    model.add(Dense(100,input_shape=(features,),activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(features,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer=Adam(lr=1e-3),metrics=['mae'])
    model.fit(train_X,train_y,batch_size=1,epochs=10,validation_data=(test_X,test_y))

    result_df = pd.DataFrame(index=test_index)
    inversed_test = test_scaler.inverse_transform(test)
    print(inversed_test.shape)
    result_df = result_df[:len(inversed_test)]
    result_df['Actuals'] = inversed_test
    final_preds = model.predict(test_X)
    print(final_preds.shape)
    zeros = np.zeros((final_preds.shape[0],preds.shape[1] - 1))
    final_preds = np.concatenate([final_preds,zeros],axis=1)
    result_df['Predictions'] = preds_scaler.inverse_transform(final_preds)[:,0]
    print(result_df.head(5))
    result_df.dropna(inplace=True)
    print(result_df.head(5))
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')

def create_24hour_predictions(dataset):
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]
    pred_df = pd.DataFrame(index=test.index)
    sequence_len = 24
    lstm_model_names = ["LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence24-1557742225.h5","CNNLSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence24-1557744808.h5","LSTM_encdec_L1L2_datasetcombined_df_stripped.pickle_sequence24-1552984831.h5","LSTM_encdec_L1L2_datasetcombined_df_stripped.pickle_sequence24-1553029113.h5"]
    for model_name in lstm_model_names:
        dataset_name = model_name.split('dataset')[1].split('pickle')[0] + "pickle"
        test_dataset = pd.read_pickle("data/" + dataset_name)
        test_dataset = test_dataset[-(24*10*7):]
        pred_df[model_name] = lstm_prediction(test_dataset,model_name,sequence_len)

    print("LSTMs completed")
    # non-lstm predictions
    prophet_df = prophet_prediction(train,test)
    pred_df['Prophet'] = prophet_df['yhat']
    print("Prophet completed")
    pred_df.dropna(inplace=True)
    pred_df['AR'] = ar_prediction(train,test,sequence_len)
    print("AR completed")
    pred_df.to_pickle('ensemble_temp_df_seq24')
    return pred_df

def create_36hour_predictions(dataset):
    """
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]
    pred_df = pd.DataFrame(index=test.index)
    #print(len(pred_df))
    sequence_len = 36
    lstm_model_names = ['LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe3_shifted.pickle_sequence36-1557145204.h5','LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence36-1557739602.h5','CNNLSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence36-1557745825.h5','LSTM_encdec_lossMAE_datasetcombined_df_stripped_T24.pickle_sequence36-1556884321.h5','CNNLSTM_encdec_lossMAE_datasetcombined_df_stripped_T24.pickle_sequence36-1556884840.h5','LSTM_encdec_lossMAE_datasetcombined_df_stripped_swe3_shifted.pickle_sequence36-1556885114.h5','CNNLSTM_encdec_lossMAE_datasetcombined_df_stripped_swe3_shifted.pickle_sequence36-1556885932.h5','LSTM_encdec_lossMAE_datasetcombined_df_stripped_swe3.pickle_sequence36-1557000181.h5','CNNLSTM_encdec_lossMAE_datasetcombined_df_stripped_swe3.pickle_sequence36-1557001514.h5','LSTM_encdec_lossMAE_datasetcombined_df_engineered_T-24.pickle_sequence36-1557003068.h5','CNNLSTM_encdec_lossMAE_datasetcombined_df_engineered_T-24.pickle_sequence36-1557003576.h5','LSTM_encdec_lossMAE_datasettwo_features_df.pickle_sequence36-1557007526.h5','CNNLSTM_encdec_lossMAE_datasettwo_features_df.pickle_sequence36-1557008755.h5']
    #lstm_model_names = ['LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe3_shifted.pickle_sequence36-1557145204.h5','CNNLSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence36-1557745825.h5']
    for model_name in lstm_model_names:
        dataset_name = model_name.split('dataset')[1].split('pickle')[0] + "pickle"
        test_dataset = pd.read_pickle("data/" + dataset_name)
        test_dataset = test_dataset[-(24*10*7):]
        columns_to_drop = [x for x in test_dataset.columns if 'Spot' and 'T-' in x and int(x.split('-')[1]) < sequence_len]
        test_dataset.drop(columns=columns_to_drop,inplace=True)
        preds = lstm_prediction(test_dataset,model_name,sequence_len)
        pred_df = pred_df[:len(preds)]
        pred_df[model_name] = lstm_prediction(test_dataset,model_name,sequence_len)
        print(f"{model_name} prediction completed")

    print("LSTMs completed")
    # non-lstm predictions
    prophet_df = prophet_prediction(train,test)
    pred_df['Prophet'] = prophet_df['yhat']
    print("Prophet completed")
    pred_df.dropna(inplace=True)
    pred_df['AR'] = ar_prediction(train,test,sequence_len)
    print("AR completed")
    pred_df.to_pickle('trying_best_36_df.pickle')
    """
    return pd.read_pickle('trying_best_36_df.pickle')

def create_48hour_predictions(dataset):
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]
    pred_df = pd.DataFrame(index=test.index)
    sequence_len = 48
    lstm_model_names = ['LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence48-1557744055.h5','CNNLSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence48-1557747123.h5']
    for model_name in lstm_model_names:
        print(model_name)
        dataset_name = model_name.split('dataset')[1].split('pickle')[0] + "pickle"
        test_dataset = pd.read_pickle("data/" + dataset_name)
        test_dataset = test_dataset[-(24*10*7):]
        columns_to_drop = [x for x in test_dataset.columns if 'Spot' and 'T-' in x and int(x.split('-')[1]) < sequence_len]
        test_dataset.drop(columns=columns_to_drop,inplace=True)
        pred_df[model_name] = lstm_prediction(test_dataset,model_name,sequence_len)
    print("LSTMs completed")
    # non-lstm predictions
    prophet_df = prophet_prediction(train,test)
    pred_df['Prophet'] = prophet_df['yhat']
    print("Prophet completed")
    pred_df.dropna(inplace=True)
    pred_df['AR'] = ar_prediction(train,test,sequence_len)
    print("AR completed")
    pred_df.to_pickle('ensemble_temp_df_seq24')
    return pred_df

def create_168hour_predictions(dataset):
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]
    pred_df = pd.DataFrame(index=test.index)
    sequence_len = 168
    #hw_df = pd.read_pickle('HW168_preds.pickle')
    #pred_df['HW'] = hw_df['HW 168']
    #print("HW completed")
    """
    lstm_model_names = ['LSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence48-1557744055.h5','CNNLSTM_encdec_stdscaler_lossMAE_datasetcombined_df_stripped_swe4_shifted.pickle_sequence48-1557747123.h5']
    for model_name in lstm_model_names:
        print(model_name)
        dataset_name = model_name.split('dataset')[1].split('pickle')[0] + "pickle"
        test_dataset = pd.read_pickle("data/" + dataset_name)
        test_dataset = test_dataset[-(24*10*7):]
        columns_to_drop = [x for x in test_dataset.columns if 'Spot' and 'T-' in x and int(x.split('-')[1]) < sequence_len]
        test_dataset.drop(columns=columns_to_drop,inplace=True)
        pred_df[model_name] = lstm_prediction(test_dataset,model_name,sequence_len)
    print("LSTMs completed")
    """
    # non-lstm predictions
    prophet_df = prophet_prediction(train,test)
    pred_df['Prophet'] = prophet_df['yhat']
    print("Prophet completed")
    pred_df.dropna(inplace=True)
    pred_df['AR'] = ar_prediction(train,test,sequence_len)
    print("AR completed")
    pred_df.to_pickle('ensemble_temp_df_seq168')
    
    return pred_df

def mean_absolute_percentage_error(true,pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true))) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,help='Filename of the results')
    parser.add_argument('--mean',action='store_true')
    parser.add_argument('--median',action='store_true')
    parser.add_argument('--nn',action='store_true')
    parser.add_argument('--rf',action='store_true')
    parser.add_argument('--sequence',type=int)
    parser.add_argument('--load',type=str)
    args = parser.parse_args()

    dataset = pd.read_pickle('data/shifted/combined_df_stripped_swe4_shifted.pickle')
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]

    if args.sequence == 24:
        pred_df = create_24hour_predictions(dataset)

    elif args.sequence == 36:
        pred_df = create_36hour_predictions(dataset)

    elif args.sequence == 48:
        pred_df = create_48hour_predictions(dataset)

    elif args.sequence == 168:
        pred_df = create_168hour_predictions(dataset)

    """
    for x in [36,48,168]:
        pred_df2 = pd.DataFrame()
        pred_df2[f'HW {x}'] = hw_prediction(train,test,x)
        pred_df2.to_pickle(f'HW{x}_preds.pickle')
    """
    #lstm_model_names = os.listdir('models/')
    #lstm_model_names = [x for x in lstm_model_names if 'LSTM_encdec_' in x and 'residual' not in x and "sequence24" in x]
    #lstm_model_names = lstm_model_names[:30]
    #lstm_model_names = ["CNNLSTM_encdec_lossMSE_datasetcombined_df_stripped_swe2_168.pickle_sequence168-1556004737.h5","CNNLSTM_encdec_lossMSE_datasetcombined_df_stripped_swe2_168.pickle_sequence168-1556003430.h5","CNNLSTM_encdec_lossMSE_datasetcombined_df_stripped_swe2_168.pickle_sequence168-1556001883.h5"]


    #print(r2_score(df['Actuals'],df['Predictions']))

    if args.mean is True:
        mean_of_predictions(test,pred_df.values,args.filename)
    
    elif args.median is True:
        median_of_predictions(test,pred_df.values,args.filename)

    elif args.nn is True:
        #neural_network_metalearner(test,pred_df.values,args.filename)
        neural_network_metalearner_wo_scaling(test,pred_df.values,args.filename)

    elif args.rf is True:
        random_forest_metalearner(test,pred_df.values,args.filename)
        
