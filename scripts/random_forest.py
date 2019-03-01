import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
import json

def create_test_forest(combined_df):
    labels = combined_df.values[:,0]
    features = combined_df.values[:,1:]
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.10,shuffle=False)
    #forest = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=-1)
    forest = RandomForestRegressor(n_estimators=600,min_samples_split=10,min_samples_leaf=4,max_features='sqrt',max_depth=90,bootstrap=False,random_state=42,n_jobs=-1)
    forest.fit(train_features, train_labels)
    preds = forest.predict(test_features)
    mae = mean_absolute_error(preds,test_labels)
    rmse = np.sqrt(mean_squared_error(preds,test_labels))
    print(f"Test MAE: {mae}")
    print(f"Test RMSE: {rmse}")
    return forest

def random_forest_random_search(combined_df,random_grid):
    labels = combined_df.values[:,0]
    features = combined_df.values[:,1:]
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.10,shuffle=False)
    forest = RandomForestRegressor(random_state=42)
    random_search_forest = RandomizedSearchCV(
                                            estimator=forest, 
                                            param_distributions=random_grid,
                                            scoring='neg_mean_absolute_error', 
                                            n_iter=25,
                                            n_jobs=-1,
                                            cv=2,
                                            verbose=2,
                                            random_state=42,
                                            return_train_score=True)
    random_search_forest.fit(train_features,train_labels)
    np.save('rf_best_params.npy',random_search_forest.best_params_)
    np.save('rf_random_search_results.npy',random_search_forest.cv_results_)
    #with open('rf_best_params','w+') as f:
    #    f.write([str(x) for x in list(random_search_forest.best_params_)])
    #with open('rf_random_search_results','w+') as ff:
    #    ff.write([str(x) for x in list(random_search_forest.cv_results_)])

def list_importances(forest,feature_names):
    importances = list(forest.feature_importances_)
    feature_importances = [(feature,round(importance,2)) for feature, importance in zip(feature_names,importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def creating_features(combined_df):
    """
    combined_df.drop(columns=['Spot MA','Spot STD','Spot T-168','Spot T-8760','Spot Rolling Min', 'Spot Rolling Max'],inplace=True)
    combined_df['Spot MA'] = combined_df['Spot'].rolling(window=168,center=False).mean()
    combined_df['Spot STD'] = combined_df['Spot'].rolling(window=168,center=False).std()
    combined_df['Spot T-168'] = combined_df['Spot'].shift(168)
    #combined_df['Spot T-8760'] = combined_df['Spot'].shift(8760)
    combined_df['Spot Rolling Min'] = combined_df['Spot'].rolling(window=168,center=False).min()
    combined_df['Spot Rolling Max'] = combined_df['Spot'].rolling(window=168,center=False).max()
    combined_df['Spot T-336'] = combined_df['Spot'].shift(336)
    combined_df['Spot T-504'] = combined_df['Spot'].shift(504)
    combined_df['Weekday'] = combined_df.index.weekday
    combined_df.dropna(inplace=True)"""
    combined_df['Spot Expanding Average'] = combined_df['Spot'].expanding().mean()
    combined_df['Spot Expanding STD'] = combined_df['Spot'].expanding().std()
    combined_df.dropna(inplace=True)
    print(combined_df.head(10))
    return combined_df

combined_df = pd.read_pickle('data/combined_df_engineered.pickle')
combined_df = creating_features(combined_df)
#combined_df.drop(columns=combined_df.columns[26:],inplace=True)
#combined_df.drop(columns=combined_df.columns[27:76],inplace=True) # drop one hot encoded days etc
print(combined_df.columns)
feature_names = combined_df.drop('Spot',axis=1).columns
forest = create_test_forest(combined_df)
random_grid = {
    'n_estimators' : [int(x) for x in np.linspace(start=200,stop=2000, num=10)],
    'max_features' : ['auto','sqrt'],
    'max_depth' : [int(x) for x in np.linspace(10,110,num=11)] + [None],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [1,2,4],
    'bootstrap' : [True,False]
}
#random_forest_random_search(combined_df,random_grid)
#best_params = np.load('rf_best_params.npy').item()
#print(best_params)
#print("-------")
#results = np.load('rf_random_search_results.npy').item()
#print(results)

joblib.dump(forest,'models/random_forest8.sav')
#forest = joblib.load('models/random_forest5.sav')

list_importances(forest,feature_names)
