# function that trains and save models
from enum import EnumMeta
from black import Report
import joblib
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import config as cfg
import data.processed as processed_data
from src.features import build_features
from src.data import make_dataset
import argparse
import sys
import os
import json
import pandas as pd
from src.data.make_dataset import  read_params_file
import data.raw as raw
import data.processed as processed_data
import saved_models
import reports
from src.models.pipeline import pipeline
from joblib import dump, load
import time
# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def make_prediction(config_path):
    model_path = os.path.dirname(saved_models.__file__)
    reports_dir = os.path.dirname(reports.__file__)

    best_model_path = os.path.join(model_path , config_path.get("best_model"))
    prediction_path = os.path.join(reports_dir , config_path.get("reports").get("prediction"))

    _, _, X_test, y_test = pipeline(config_path=parse_args.config)

    model = joblib.load(best_model_path)

    online_data_x = X_test.head(5)
    online_data_y = y_test.head(5)

    result = model.predict(online_data_x)

    prediction = {}

    for index, val in enumerate(result):
        if(val == 1):
            prediction[f"prediction_{index}"] = "Approve"
        else:
            prediction[f"prediction_{index}"]= "Reject"

    with open(prediction_path, "w+") as f:
        json.dump(prediction, f, indent=4)

    
    print(prediction)
    










if __name__ == '__main__':
    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()

    make_prediction(config_path=parse_args.config)
