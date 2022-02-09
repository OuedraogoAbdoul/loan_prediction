# # download or generate data
import pandas as pd
import argparse
import yaml
import os



def get_main_directory_path():
    return os.path.dirname(__file__).split("src")[0]



def read_params_file(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config



def download_data(config_path):
    data_path = read_params_file(config_path)
    df_data = pd.read_csv(data_path["load_data"]["row_dataset_link"])

    df_data.to_csv(os.path.join(get_main_directory_path(), data_path["load_data"]["raw_dataset_csv"]))






def load_datasets():
    pass



def split_datasets():
    pass


    



if __name__ =='__main__':
    config_yml_path = os.path.join(get_main_directory_path(), "config/params.yaml") 
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config_yml_path)
    parse_args = parser.parse_args()

    data = download_data(config_path=config_yml_path)
    
