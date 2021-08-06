"""
main module to train house price prediction model
"""
import pandas as pd
from load_data import load_raw_data
from trainer import train
from feature_preprocess import preprocess_features
import sys
from model_registery import register_model
if __name__ == '__main__':
    #Read arguments
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    max_features = int(sys.argv[3])
    reg_model = sys.argv[4]

    #Prep parameter dictionary
    param_dict = {}
    param_dict['n_estimators'] = n_estimators
    param_dict['max_depth'] = max_depth
    param_dict['max_features'] = max_features

    #Load Training data
    load_raw_data()

    #Model Training
    train_data = pd.read_csv('data/train.csv')
    processed_data = preprocess_features(train_data)
    y = train_data['price'].values
    train(processed_data, y, param_dict)

    #Register Model
    if reg_model == "T":
        register_model()

