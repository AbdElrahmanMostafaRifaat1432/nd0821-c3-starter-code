import os
import pytest
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Import from your own modules:
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the cleaned Census dataset.
    """
    csv_path = "./starter/data/census_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    return df

@pytest.fixture(scope="session")
def processed_data(data):
    """
    Fixture to process the dataset, splitting into X, y with the provided cat features.
    Returns X, y, encoder, lb.
    """
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
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    return X, y, encoder, lb

def test_data_shape(data):
    """
    Ensure the imported DataFrame is not empty.
    """
    assert data.shape[0] > 0, "Loaded data has 0 rows."
    assert data.shape[1] > 0, "Loaded data has 0 columns."

def test_process_data(processed_data):
    X, y, encoder, lb = processed_data
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

def test_train_model(processed_data):
    X, y, encoder, lb = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

    # try to infer on the first 5 samples
    sample_X = X[:5]
    preds = inference(model, sample_X)
    assert len(preds) == 5, "Number of predictions should doesnot match the number of samples"

    def test_compute_model_metrics(processed_data):
        X, y, _, _ = processed_data
        preds = np.random.randint(0, 2, len(X))
        precision, recall, fbeta = compute_model_metrics(y, preds)
        assert 0.0 <= precision <= 1.0, "Precision out of range [0, 1]."
        assert 0.0 <= recall <= 1.0,    "Recall out of range [0, 1]."
        assert 0.0 <= fbeta <= 1.0,    "F-beta out of range [0, 1]."