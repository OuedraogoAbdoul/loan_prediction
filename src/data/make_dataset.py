# # download or generate data
from email.policy import default
from numpy import size
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import yaml
import os
import config as cfg
from data import raw



def join_paths(module_path, file_name):

    return os.path.join(os.path.join(os.path.dirname(module_path.__file__), f"{file_name}"))


def read_params_file(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config



def download_data(config_path):
    file_path = config_path["load_data"]["row_dataset_link"]
    df_data = pd.read_csv(file_path)
    

    df_data.to_csv(join_paths(raw, config_path["load_data"]["raw_dataset_csv"]))
    return df_data



def load_datasets(config_path):
    file_path = join_paths(raw, config_path["load_data"]["raw_dataset_csv"])
    return pd.read_csv(file_path)
    

def split_datasets(config_path):
    df = load_datasets(config_path)

    X_train, X_test, y_train, y_test  = train_test_split(df.drop(columns=config_path["base_model_params"]["target"]), df[config_path["base_model_params"]["target"]], test_size=config_path["split_data"]["test_size"], random_state=config_path["base_model_params"]["random_state"])
    
    X_train.to_csv(os.path.join(os.path.join(os.path.dirname(raw.__file__), config_path["split_data"]["X_train_path"])))
    y_train.to_csv(os.path.join(os.path.join(os.path.dirname(raw.__file__), config_path["split_data"]["y_train_path"])))

    X_test.to_csv(os.path.join(os.path.join(os.path.dirname(raw.__file__), config_path["split_data"]["X_test_path"])))
    y_test.to_csv(os.path.join(os.path.join(os.path.dirname(raw.__file__), config_path["split_data"]["y_test_path"])))

    return X_train, y_train, X_test, y_test
    


if __name__ =='__main__':

    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()

    data = download_data(config_path=parse_args.config)
    X_train, y_train, X_test, y_test = split_datasets(config_path=parse_args.config)
    
