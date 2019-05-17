import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
from rfpimp import importances
import itertools

def mean_absolute_percentage_error(true,pred):
    return np.mean(np.abs((np.array(true) - np.array(pred)) / np.array(true))) * 100

def extra_forest(combined_df):
    labels = combined_df.values[:,0]
    features = combined_df.values[:,1:]
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.10,shuffle=False)
    forest = RandomForestRegressor(n_estimators=100,
                                   min_samples_split=12,
                                   min_samples_leaf=5,
                                   max_features='sqrt',
                                   max_depth=60,
                                   bootstrap=False,
                                   random_state=42,
                                   n_jobs=-1,
                                   criterion='mae')
    forest.fit(train_features, train_labels)
    preds = forest.predict(test_features)
    mae = mean_absolute_error(preds,test_labels)
    rmse = np.sqrt(mean_squared_error(preds,test_labels))
    print(f"Test MAE: {mae}")
    print(f"Test RMSE: {rmse}")
    return forest
    

def random_forest_random_search(train_x,train_y):
    random_grid = {
    'n_estimators' : [int(x) for x in np.linspace(start=100,stop=1000, num=11)],
    'max_features' : ['auto','sqrt'],
    'max_depth' : [int(x) for x in np.linspace(4,11,num=8)] + [None],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [1,2,4],
    'bootstrap' : [True,False]
    }
    forest = RandomForestRegressor(random_state=42)
    random_search_forest = RandomizedSearchCV(
                                            estimator=forest, 
                                            param_distributions=random_grid,
                                            scoring='neg_mean_absolute_error', 
                                            n_iter=50,
                                            n_jobs=-1,
                                            cv=2,
                                            verbose=2,
                                            random_state=42,
                                            return_train_score=True)
    random_search_forest.fit(train_x,train_y)
    print(random_search_forest.best_params_)
    np.save('rf_random_search_best_params2.npy',random_search_forest.best_params_)
    np.save('rf_random_search_results2.npy',random_search_forest.cv_results_)

def random_forest_grid_search(train_x,train_y):
    grid = {
        'n_estimators' : [200,300,400],
        'min_samples_split' : [12],
        'min_samples_leaf' : [4],
        'max_features' : ['sqrt',None],
        'max_depth' : [10,30,50],
        'bootstrap' : [False]
    }
    forest = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator = forest, 
                               param_grid=grid,
                               cv=2,
                               n_jobs=-1,
                               verbose=2,
                               return_train_score=True)
    grid_search.fit(train_x,train_y)
    print(grid_search.best_params_)

def list_importances(forest,feature_names):
    importances = list(forest.feature_importances_)
    feature_importances = [(feature,round(importance,2)) for feature, importance in zip(feature_names,importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse = True)
    feature_importances_df = pd.DataFrame(data=feature_importances,columns=['Feature','Importance Score'])
    feature_importances_df.set_index('Feature',inplace=True)
    print(feature_importances_df)
    ax = feature_importances_df.plot.barh()
    plt.show(ax)
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def list_permutation_importances(forest,test):
    test = test[-168*4:]
    permutative_importances = importances(forest,test.drop(columns=['Spot']),test['Spot'])
    #fig, ax = plt.subplots()
    ax = permutative_importances.plot.barh()
    plt.savefig('../images/permutative_importances.png')
    plt.show(ax)

def predict_forest(forest,dataset):
    labels = dataset.values[:,0]
    features = dataset.values[:,1:]
    predictions = forest.predict(features)
    mae = mean_absolute_error(predictions,labels)
    return predictions, labels

def forecast_t24():
    combined_df = pd.read_pickle('data/shifted/combined_df_stripped_swe4_shifted.pickle')
    t24_df = pd.read_pickle('data/shifted/combined_df_stripped_T24.pickle')
    columns_to_drop = ['Spot MA T-24','Spot STD T-24','Spot Rolling Min T-24','Spot Rolling Max T-24']
    combined_df.drop(columns=columns_to_drop,inplace=True)
    combined_df2 = combined_df['2015-02':'2018-12']
    columns_to_shift = [x for x in combined_df2.columns if 'Spot' not in x]
    combined_df2[columns_to_shift] = combined_df2[columns_to_shift].shift(24)
    combined_df.drop(columns=columns_to_shift,inplace=True)
    combined_df[columns_to_shift] = combined_df2[columns_to_shift]
    combined_df['Spot T-168'] = t24_df['Spot T-168']
    combined_df.dropna(inplace=True)
    train_x, train_y = combined_df.drop(columns='Spot')[:-(24*10*7)], combined_df['Spot'][:-(24*10*7)]
    test_x, test_y = combined_df.drop(columns='Spot')[-(24*10*7):], combined_df['Spot'][-(24*10*7):]
    test_df = pd.DataFrame(index=test_y.index, data={'Actual' : test_y})

    #random_forest_random_search(train_x,train_y)
    #random_forest_grid_search(train_x,train_y)
    

    forest = RandomForestRegressor(n_estimators=200,
                                   min_samples_split=12,
                                   min_samples_leaf=4,
                                   max_features='sqrt',
                                   max_depth=12,
                                   bootstrap=False,
                                   random_state=42,
                                   n_jobs=-1)
    forest.fit(train_x, train_y)
    preds = forest.predict(test_x)
    mae = mean_absolute_error(preds,test_y)
    rmse = np.sqrt(mean_squared_error(preds,test_y))
    mape = mean_absolute_percentage_error(preds,test_y)
    print(f"Test MAE: {mae}")
    print(f"Test RMSE: {rmse}")
    print(f"Test MAPE: {mape}")
    
    feature_names = combined_df.drop('Spot',axis=1).columns
    list_importances(forest,feature_names)
    list_permutation_importances(forest,combined_df)
    
if __name__ == '__main__':
    forecast_t24()