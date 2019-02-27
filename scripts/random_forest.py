import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals import joblib

def create_test_forest(combined_df):
    labels = combined_df.values[:,0]
    features = combined_df.values[:,1:]
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.10,shuffle=False)
    forest = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs=-1)
    forest.fit(train_features, train_labels)
    preds = forest.predict(test_features)
    mae = mean_absolute_error(preds,test_labels)
    rmse = np.sqrt(mean_squared_error(preds,test_labels))
    print(f"Test MAE: {mae}")
    print(f"Test RMSE: {rmse}")
    return forest

def list_importances(forest,feature_names):
    importances = list(forest.feature_importances_)
    feature_importances = [(feature,round(importance,2)) for feature, importance in zip(feature_names,importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse = True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def editing_df(combined_df):
    combined_df.drop(columns=['Spot MA','Spot STD','Spot T-168','Spot T-8760','Spot Rolling Min', 'Spot Rolling Max'],inplace=True)
    combined_df['Spot MA'] = combined_df['Spot'].rolling(window=168,center=False).mean()
    combined_df['Spot STD'] = combined_df['Spot'].rolling(window=168,center=False).std()
    combined_df['Spot T-168'] = combined_df['Spot'].shift(168)
    #combined_df['Spot T-8760'] = combined_df['Spot'].shift(8760)
    combined_df['Spot Rolling Min'] = combined_df['Spot'].rolling(window=168,center=False).min()
    combined_df['Spot Rolling Max'] = combined_df['Spot'].rolling(window=168,center=False).max()
    combined_df['Spot T-336'] = combined_df['Spot'].shift(336)
    combined_df['Spot T-504'] = combined_df['Spot'].shift(504)
    combined_df.dropna(inplace=True)
    print(combined_df.head(10))
    return combined_df

combined_df = pd.read_pickle('data/combined_df_engineered.pickle')
#combined_df['Spot T-336'] = combined_df['Spot'].shift(336).bfill()
#combined_df['Spot T-504'] = combined_df['Spot'].shift(504).bfill()
# poista bfill ja kokeile toimiiko -> jos ei niin dropna
combined_df = editing_df(combined_df)

feature_names = combined_df.drop('Spot',axis=1).columns
forest = create_test_forest(combined_df)
joblib.dump(forest,'models/random_forest6.sav')
#forest = joblib.load('models/random_forest4.sav')
list_importances(forest,feature_names)
