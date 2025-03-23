# run the following command 
# uvicorn starter.app:app --host 0.0.0.0 --port 8000 --reload
# Put the code for your API here.
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference
import os

# Load model and encoders
model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
lb = joblib.load("starter/model/lb.pkl")

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

app = FastAPI(title="Census Inference API",
    description="This API provides a greeting at the root (GET) and model inference at the /predict endpoint (POST).",
    version="1.0")

# Pydantic model for request body
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., alias="marital-status",example="Never-married") # fix them using alias 
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: Literal["Male", "Female"] = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(...,alias="native-country" , example="United-States")

class Config:
        populate_by_name = True  # Allows FastAPI to use the alias when parsing JSON


@app.get("/")
def read_root() -> dict:
    
    return {"message": "Welcome to the Model API!"}

@app.post("/predict")
def predict(data: CensusData) -> dict:
    """Perform model inference."""
    # Convert input to DataFrame
    input_data = pd.DataFrame([data.model_dump(by_alias=True)])

    # input_data.rename(columns={"marital-status": "marital_status", "native-country": "native_country"}, inplace=True)


    # Process the data
    X, _, _, _ = process_data(input_data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    # Run inference
    pred = inference(model, X)

    # Convert prediction to readable format
    prediction_label = lb.inverse_transform(pred)[0]

    return {"prediction": prediction_label}
