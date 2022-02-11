# function that trains and save models
from black import Report
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
import joblib
import time
# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def evaluation_metrics(ground_truth, pred):

    # ROC
    roc_auc = roc_auc_score(ground_truth, pred)
    # fpr, tpr, _ = roc_curve(ground_truth, pred)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    # Confusion matrix
    # cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # disp.plot()

    # balance accuracy
    balanced_accuracy_metric = balanced_accuracy_score(ground_truth, pred)



    # precision and recall
    # precision_score(y_true, y_pred, average=None)
    precision, recall, fbeta_score, _ = precision_recall_fscore_support(ground_truth, pred, average="binary")
    # display = PrecisionRecallDisplay.from_estimator(
    # classifier, X_test, y_test, name="LinearSVC")_ = display.ax_.set_title("2-class Precision-Recall curve")
    
    return balanced_accuracy_metric, roc_auc, precision, recall, fbeta_score



def model_experiment(config_path):
    model_dir = os.path.dirname(saved_models.__file__)
    reports_dir = os.path.dirname(reports.__file__)
    is_best_model = config_path.get("best_model")
    metrics_data = []

    scores_file = os.path.join(reports_dir ,config_path.get("reports").get("scores"))

    params_file = config_path.get("reports").get("params")
    random_state = config_path.get("base_model_params").get("random_state")
    
    if(not is_best_model):
        models = [LogisticRegression(random_state=random_state, max_iter=200), DecisionTreeClassifier(max_depth=5, min_samples_split=2)]
    else:
        models = [svm.SVC()]
    
    X_train, y_train, X_test, y_test = pipeline(config_path=parse_args.config)
    
    for model in models:

        print(f"Training {model}")
        start = time.time()
        clf = model.fit(X_train, y_train)
        stop = time.time()
        print(f"Evaluating {model}".split("(")[0])
        pred = clf.predict(X_test)

        # Save model
        if(not is_best_model):
            model_name = f"{model}".split("(")[0]
        else:
            model_name = config_path.get("best_model_name")

        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        
        

        joblib.dump(clf, model_path)

        # Save evaluation metrics
        balanced_accuracy_metric, roc_auc, precision, recall, fbeta_score = evaluation_metrics(y_test, pred)

        scores = {

            f"{model}".split("(")[0]: {
                "balanced_accuracy_metric": balanced_accuracy_metric,
                "precision": precision,
                "recall": recall,
                "fbeta_score": fbeta_score,
                "roc_auc": roc_auc,
                "train_time(s)": stop - start
            }
            
        }


        metrics_data.append(scores)


    with open(scores_file, "w+") as f:
        json.dump(metrics_data, f, indent=4)


    # with open(params_file, "w") as f:
    #     params = {
    #         "alpha": alpha,
    #         "l1_ratio": l1_ratio,
    #     }
    #     json.dump(params, f, indent=4)



        
def run_training(config_path):

    model_experiment(config_path)
    # pass
    


if __name__ == '__main__':
    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()

    run_training(config_path=parse_args.config)
