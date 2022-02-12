from typing import Optional
import os
import argparse
from fastapi import FastAPI,  Request
import pandas as pd
import json
import joblib
from src.features.build_features import feature_eng
from src.models.predict_model import make_prediction
from src.data.make_dataset import  read_params_file
from fastapi.responses import HTMLResponse


# from pydantic import BaseModel


# class Item(BaseModel):
#     name: str
#     description: Optional[str] = None
#     price: float
#     tax: Optional[float] = None


app = FastAPI()

@app.get("/")
def root(request: Request):
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h3>Welcome to our loan acceptance and approval tool! This is an experiment and should only be used as such.</h3>"
        "<div>"
        "Click on the docs: <a href='/docs'>here </a>"
        "</div>"
        
        "</html>"



    )

    return HTMLResponse(content=body)


# src="serving_loan_prediction_model/requirements/cover.jpg">


@app.post("/predict")
async def predict():
    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    prediction, features = make_prediction(config_path=config)



    return prediction, features


if __name__ == '__main__':
    PWD = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(PWD, '../..'))

    param_file = os.path.join(ROOT, "params.yaml")

    config = read_params_file(param_file)
    
    parser = argparse.ArgumentParser(description="Take data path")
    parser.add_argument("--config", default=config)

    parse_args = parser.parse_args()

    make_prediction(config_path=parse_args.config)
