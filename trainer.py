"Module to train a Model to predict price of a house"
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from feature_preprocess import preprocess_features
import pandas as pd
from mlflow import log_param, log_metric
from mlflow.sklearn import log_model
from mlflow.models.signature import infer_signature
import pandas as pd

def train(X, y, param_dict):
    """
    This method will train a model to predict house price
    :param X: Training Features
    :param y: Target Data
    :return: None
    """
    #Read hyperparameter for Training
    #param_dict = {'n_estimators':40, 'max_depth':4, 'max_features':7}
    n_estimators = param_dict['n_estimators']
    max_depth = param_dict['max_depth']
    max_features = param_dict['max_features']

    #Train
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                     random_state=1234)
    rf_model.fit(X, y)

    #Test and Validation Data
    val_data = pd.read_csv('data/val.csv')
    test_data = pd.read_csv('data/test.csv')

    #Evaluate Model
    train_r2_score = r2_score(y, rf_model.predict(X))
    val_r2_score = r2_score(val_data['price'].values, rf_model.predict(preprocess_features(val_data)))
    test_r2_score = r2_score(test_data['price'].values, rf_model.predict(preprocess_features(test_data)))

    train_mse = mean_squared_error(y, rf_model.predict(X))
    val_mse = mean_squared_error(val_data['price'].values, rf_model.predict(preprocess_features(val_data)))
    test_mse = mean_squared_error(test_data['price'].values, rf_model.predict(preprocess_features(test_data)))

    #Log hyperparameter for training
    log_param("n_estimators", n_estimators)
    log_param('max_depth', max_depth)
    log_param('max_features',max_features)

    #Log Metrics
    log_metric('train_r2_score', train_r2_score)
    log_metric('val_r2_score', val_r2_score)
    log_metric('test_r2_score', test_r2_score)
    log_metric('train_mse', train_mse)
    log_metric('val_mse', val_mse)
    log_metric('test_mse', test_mse)

    #Infer signature
    signature = infer_signature(X, rf_model.predict(X))

    #log model
    log_model(rf_model, "house_price_model", signature=signature)

    return None