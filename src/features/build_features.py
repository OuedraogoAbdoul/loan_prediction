# process raw data into features for modeling

# step to peform:
    # load data
    # transform variables
    # encode categorical variable
    # impute missig values
    # use imbalance library to sample the data
    # scale the data

# import the libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn  as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
import category_encoders as ce
import argparse
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# import modules
from src.data.make_dataset import read_params_file, load_datasets
import config as cfg
from data import raw, processed



def get_data(config_path):
    return load_datasets(config_path)


def extract_year_from_date(df, config_path):

    col = config_path["categorical_vars"].get("earliest_cr_line")
    df[col] = pd.DatetimeIndex(df[col]).year

    return df


def extract_zip_codes(df, config_path):

    col = config_path["categorical_vars"].get("zip_code")
    df.zip_code = df.zip_code.str[:3]

    return df




def encode_data(df, config_path):
    col = config_path["numercial_vars"].get("is_bad")

    encoder = TargetEncoder(True, handle_missing='missing', handle_unknown='missing')
    df = encoder.fit_transform(df, df[col])

    return df


def impute_missing_values(df, config_path):

    df.columns.tolist()
    imputer = SimpleImputer(strategy='median', fill_value = 'Missing')
    imputer.fit(df)

    tmp_data = pd.DataFrame(imputer.transform(df), columns=df.columns.tolist())

    tmp_data.replace("NaN", "Missing", inplace=True)


    return tmp_data


def scale_variables(df, config_path):

    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns.to_list())

    return scaled_data



def solve_imbalance_data(df, config_path):
    col = config_path["numercial_vars"].get("is_bad")

    sm = SMOTE(
    sampling_strategy='auto',  
    random_state=42, 
    k_neighbors=5,
    n_jobs=4
    )

    X_sm, y_sm = sm.fit_resample(df.drop(columns=[col]), df[col])


    tl = TomekLinks(
        sampling_strategy='all',
        n_jobs=4)

    smtomek = SMOTETomek(
        sampling_strategy='auto',  
        random_state=42, 
        smote=sm,
        tomek=tl,
        n_jobs=4
    )

    X_smtl, y_smtl = smtomek.fit_resample(df.drop(columns=[col]), df[col])

    X_train, X_test, y_train, y_test  = train_test_split(X_smtl, y_smtl, test_size=config_path["split_data_processed"]["test_size"], random_state=config_path["base_model_params"]["random_state"])
    
    X_train.to_csv(os.path.join(os.path.join(os.path.dirname(processed.__file__), config_path["split_data_processed"]["X_train_path"])))
    y_train.to_csv(os.path.join(os.path.join(os.path.dirname(processed.__file__), config_path["split_data_processed"]["y_train_path"])))

    X_test.to_csv(os.path.join(os.path.join(os.path.dirname(processed.__file__), config_path["split_data_processed"]["X_test_path"])))
    y_test.to_csv(os.path.join(os.path.join(os.path.dirname(processed.__file__), config_path["split_data_processed"]["y_test_path"])))


    return X_train, y_train, X_test, y_test

def feature_eng(config_path):

    df = get_data(config_path)

    df = extract_year_from_date(df, config_path)

    df = extract_zip_codes(df, config_path)

    df = encode_data(df, config_path)

    df = impute_missing_values(df, config_path)
    
    df = scale_variables(df, config_path)

    X_train, y_train, X_test, y_test = solve_imbalance_data(df, config_path)


    # print(data.head())
    return X_train, y_train, X_test, y_test

if __name__ =='__main__':
    
    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()
    X_train, y_train, X_test, y_test  = feature_eng(config_path=parse_args.config)


    

    
