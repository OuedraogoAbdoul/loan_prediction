from data import raw
import config as cf
import os
from os.path import exists
from src.data.make_dataset import read_params_file
import pytest



param_path = os.path.join(os.path.dirname(cf.__file__), "params.yaml")
config = read_params_file(param_path)

data_path = os.path.join(os.path.dirname(raw.__file__), config["load_data"]["raw_dataset_csv"])


def test_data_exist():
    
    assert(exists(data_path) == True, "Passed test")