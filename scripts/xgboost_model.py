import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import xgboost
import os


def xgboost_forecast(history,sequence_len,model):
    history = np.array(history)
    model.fit(history).fit()
    yhat = model.predict(history)
    return yhat

def xgboost_prediction(train,test,sequence_len):
    train = train
    test = test['Spot']
    #test = pd.DataFrame(data=test_data['FI'].values,index=test_data.index)
    # training data 2018
    predictions_xg = list()
    history_xg = [x for x in train]
    hour_counter = sequence_len - 1 # 24-1 to prevent index out of bounds
    for i in tqdm(range(int(len(test)/sequence_len))):
        for _ in range(sequence_len):
            yhat = xgboost_forecast(history_xg,sequence_len)
            predictions_xg.append(yhat)
            

        for yhat in yhat_list:
            predictions_xg.append(yhat)
        for x in range(hour_counter-sequence_len,hour_counter):
            history_xg.append(test.iloc[x])
        #history.append(test[hour_counter] for hour_counter in range(hour_counter,hour_counter+24))
        hour_counter += sequence_len # 24 hour timestep
    return np.array(predictions_ar)

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

def xgboost_model(train,test,filename):
    train_X, train_y = train['Spot'], test['Spot']
    test_X, test_y = predictions[-int(len(predictions)/2):], test[-int(len(test)/2):]
    
    test = test[-int(len(test)/2):]
    #grid_search_xgboost(test.drop(columns=['Spot']),test['Spot'])
    #grid_search_xgboost(train_X,train_y['Spot'])

    xgbregressor = xgboost.XGBRegressor(objective='reg:linear',
                                       min_child_weight=7,
                                       colsample_bytree=0.9,
                                       learning_rate=0.1,
                                       max_depth=2,
                                       subsample=0.5,
                                       n_estimators=35
                                       )
    xgbregressor.fit(train_X,train_y['Spot'])
    result_df = pd.DataFrame(index=test.index)
    result_df['Actuals'] = test_y['Spot']
    result_df['Predictions'] = xgbregressor.predict(test_X)
    result_df.to_csv(f'ensemble_learning_models/test_results/{filename}',index_label='Date')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,help='Filename of the results')
    args = parser.parse_args()

    model = xgboost.XGBRegressor(objective='reg:linear',colsample_bytree=0.7,learning_rate=0.1,max_depth=2,alpha=10,n_estimators=50,subsample=0.9)


    dataset = pd.read_pickle('combined_df_stripped_swe2.pickle')
    train, test = dataset[:-(24*10*7)], dataset[-(24*10*7):]
    sequence_len = 24
    xgboost_model(train,test,args.filename)