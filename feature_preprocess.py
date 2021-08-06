"""
Module to preprocess raw data
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def preprocess_features(data):
    """
    Method to preprocess data
    :return: processed data
    """
    #Training data to create master data like min max scaler
    train_data = pd.read_csv('data/train.csv')

    #Define Feature Columns
    FEATURE_COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
       'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    #Define Min Max scaler
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data[FEATURE_COLUMNS])

    #Preprocess Input Data
    data = data[FEATURE_COLUMNS]
    data = min_max_scaler.transform(data)
    return data