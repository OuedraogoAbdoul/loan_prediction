# build this pipeline using sklean pipeline
import argparse
import os
import sklearn
from numpy import size
import pandas as pd
from src.data.make_dataset import download_data, split_datasets, read_params_file
from src.features.build_features import feature_eng
import config as cfg
import data.raw as raw
import data.processed as processed_data



def download_and_store_data(config_path):
    is_download = config_path["load_data"]["download_data"]

    if(is_download):
        data = download_data(config_path)

    X_train, y_train, X_test, y_test = split_datasets(config_path)

    return X_train, y_train, X_test, y_test


def performe_feature_eng(config_path):

    X_train, y_train, X_test, y_test  = feature_eng(config_path)

    return  X_train, y_train, X_test, y_test
    

def pipeline(config_path):

    data = download_and_store_data(config_path)
    X_train, y_train, X_test, y_test = performe_feature_eng(config_path)

    return X_train, y_train, X_test, y_test




if __name__=='__main__':
    param_file = os.path.join(os.path.dirname(cfg.__file__), "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()

    x_train, y_train, X_test, y_test = pipeline(config_path=parse_args.config)