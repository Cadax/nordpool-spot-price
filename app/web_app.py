from flask import Flask, render_template, request
#from flask import app
import json
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
#from tensorflow.python.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from tensorflow.python.keras import backend as K
import os

app = Flask(__name__)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

@app.route("/",methods=['GET'])
def frontpage():
    list_of_models = get_models()
    ensemble_list_of_models = get_ensemble_models()
    return render_template('index.html',models=list_of_models,ensemble_models=ensemble_list_of_models)

@app.route("/modelToGraph",methods=['POST'])
def graph():
    list_of_models = get_models()
    ensemble_list_of_models = get_ensemble_models()
    if "modelSelect" in request.form:
        model_name = request.form['modelSelect']
        df = pd.read_csv(f'../scripts/test_results/{model_name}')
    elif "ensembleModelSelect" in request.form:
        model_name = request.form['ensembleModelSelect']
        df = pd.read_csv(f'../scripts/ensemble_learning_models/test_results/{model_name}')
    else:
        return "RIP"
    mae = mean_absolute_error(df['Actuals'],df['Predictions'])
    rmse = np.sqrt(mean_squared_error(df['Actuals'],df['Predictions']))
    mape = mean_absolute_percentage_error(df['Predictions'],df['Actuals'])
    r_squared = r2_score(df['Actuals'],df['Predictions'])
    graph1 = go.Scatter(x=df['Date'],y=df['Actuals'],name='Actual')
    graph2 = go.Scatter(x=df['Date'],y=df['Predictions'],name='Forecast')
    graphJSON = json.dumps([graph1,graph2], cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('graph.html',graphJSON=graphJSON,model_name=model_name,models=list_of_models,ensemble_models=ensemble_list_of_models,mae=mae,rmse=rmse,mape=mape,r_squared=r_squared)

@app.route("/ensembleModelToGraph",methods=['POST'])
def graph_ensemble():
    list_of_models = get_models()
    ensemble_list_of_models = get_ensemble_models()
    model_name = request.form['ensembleModelSelect']
    df = pd.read_csv(f'../scripts/ensemble_learning_models/test_results/{model_name}')
    mae = mean_absolute_error(df['Actuals'],df['Predictions'])
    rmse = np.sqrt(mean_squared_error(df['Actuals'],df['Predictions']))
    mape = mean_absolute_percentage_error(df['Predictions'],df['Actuals'])
    r_squared = r2_score(df['Actuals'],df['Predictions'])
    graph1 = go.Scatter(x=df['Date'],y=df['Actuals'],name='Actual')
    graph2 = go.Scatter(x=df['Date'],y=df['Predictions'],name='Forecast')
    graphJSON = json.dumps([graph1,graph2], cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('graph.html',graphJSON=graphJSON,model_name=model_name,models=list_of_models,ensemble_models=ensemble_list_of_models,mae=mae,rmse=rmse,mape=mape,r_squared=r_squared)

@app.route("/da",methods=['GET','POST'])
def da():
    if request.method == 'GET':
        df = pd.read_pickle('../scripts/data/combined_df_stripped_swe.pickle')
        list_of_features = df.columns
        return render_template('da.html',features=list_of_features)
    
    elif request.method == 'POST':
        features = request.form.getlist('check')
        features = [str(x) for x in features]
        df = pd.read_pickle('../scripts/data/combined_df_stripped_swe.pickle')
        df2 = df[features].copy()
        df2.set_index(df.index,inplace=True)
        #df2 = (df2-df2.mean())/df2.std()
        df2 = (df2-df2.min())/(df2.max()-df2.min())
        graphs = []
        for col in df2.columns:
            graph = go.Scatter(x=df2.index,y=df2[col],name=col)
            graphs.append(graph)
        graphJSON = json.dumps(graphs,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('da.html',graphJSON=graphJSON)

    return render_template('da.html')
    
@app.route("/getModel", methods=['POST'])
def get_model():
    if request.method == 'POST':
        #model = request.form['model']
        if model == 'lstm':
            print("lol")
            """
            results = get_models()
            return render_template('index.html',list_of_models=results)
            graphJSON = graphLSTM()
            return render_template('graph.html',
                                    graphJSON=graphJSON,
                                    model_name="LSTM")
            """
        elif model == 'cnnlstm':
            print("lol")
            """
            results = get_models('CNNLSTM')
            graphJSON = graphCNNLSTM()
            return render_tdebug_mode = int(env.get("DEBUG_MODE",1)).html',
                        gradebug_mode = int(env.get("DEBUG_MODE",1))ON,
                        moddebug_mode = int(env.get("DEBUG_MODE",1))STM")
            """
        elif model == 'hw':
            print("lol")
        elif model == 'ar':
            print("lol")


def get_models():
    path = '../scripts/test_results/'
    all_results = os.listdir(path)
    all_results = sorted(all_results, key = lambda x: os.path.getmtime(path + x))
    return all_results

def get_ensemble_models():
    path = '../scripts/ensemble_learning_models/test_results/'
    all_results = os.listdir(path)
    all_results = sorted(all_results, key = lambda x: os.path.getmtime(path + x))
    return all_results

def mean_absolute_percentage_error(x,y):
    return np.mean(np.abs((x - y) / x)) * 100

"""
def graphLSTM():
    sequence_len = 24
    model_name = "LSTM_encdec_L1L2_datasetcombined_df_stripped.pickle_sequence24-1552984831.h5"
    test_data = pd.read_pickle('../notebooks/combined_df_stripped.pickle')
    test_df = test_data[-(24*10*7):]
    dates = test_df.index
    test_result_df = make_test(test_df,model_name,sequence_len)
    graph_data1 = go.Scatter(x=dates,y=test_result_df['Actuals'],name='Actual')
    graph_data2 = go.Scatter(x=dates,y=test_result_df['Predictions'],name='Forecast')
    graph_data = [graph_data1,graph_data2]
    graphJSON = json.dumps(graph_data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def graphCNNLSTM():
    sequence_len = 24
    model_name = "1553774283.h5"
    test_data = pd.read_pickle('../notebooks/combined_df_stripped_swe3_residual.pickle')
    test_df = test_data[-(24*10*7):]
    dates = test_df.index
    test_result_df = make_test(test_df,model_name,sequence_len)
    graph_data1 = go.Scatter(x=dates,y=test_result_df['Actuals'],name='Actual')
    graph_data2 = go.Scatter(x=dates,y=test_result_df['Predictions'],name='Forecast')
    graph_data = [graph_data1,graph_data2]
    graphJSON = json.dumps(graph_data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def graphHW():
    return ""

def graphAR():
    return ""

def make_prediction(model,history,sequence_len):
    input_data = np.array(history)
    input_data = input_data.reshape((input_data.shape[0]*input_data.shape[1],input_data.shape[2]))
    input_X = input_data[-sequence_len:,:]
    yhat = model.predict(input_X.reshape((1,input_X.shape[0],input_X.shape[1])),verbose=0)
    return yhat[0]


def make_test(dataset,model_name,sequence_len):
    dates = dataset.index #remember datetimes from index
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
    print(len(norm_values))
    norm_values_weeklyseq = np.array(np.split(norm_values,len(norm_values)/sequence_len))
    # to weekly sequence
    #test = norm_values_weeklyseq[:int(len(norm_values_weeklyseq)/2)]
    #actual = norm_values_weeklyseq[int(len(norm_values_weeklyseq)/2):]
    test = norm_values_weeklyseq
    actual = norm_values_weeklyseq
    print(len(test),len(actual))
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

    test_result_df.set_index(dates,inplace=True)
    #test_result_df.to_csv(f'../scripts/test_results/{model_name}_results.csv',index=True)
    K.clear_session()
    return test_result_df
"""
