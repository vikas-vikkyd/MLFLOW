"""
Module to get Raw data and save it as train test and val
"""
#import
import mlflow
import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

def load_raw_data():
    """
    Method to get train test and validation data
    """
    #Tracking uri to save Data Artifacts
    tracking_uri = mlflow.tracking.get_tracking_uri()

    #Load Dataset
    dataset = load_boston()
    feature_names = dataset['feature_names']
    data = dataset['data']
    target = dataset['target']
    df_data = pd.DataFrame(data=data, columns=feature_names)
    df_data['price'] = target

    #split data into train, val and test
    train_df, val_df = train_test_split(df_data, test_size=0.3)
    val_df, test_df = train_test_split(val_df, test_size=0.5)

    #Save Train, val and Test Data
    if not os.path.exists("data"):
        os.mkdir("data")
    with open("data/train.csv", "w") as f:
        train_df.to_csv(f)
    f.close()
    with open("data/val.csv", "w") as f:
        val_df.to_csv(f)
    f.close()
    with open("data/test.csv", "w") as f:
        test_df.to_csv(f)
    f.close()

    #Log Artifacts
    mlflow.log_artifacts("data")
    return None