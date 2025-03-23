# run the following command
# python -m pytest /mnt/d/mlops/course4/my_work/nd0821-c3-starter-code/starter/tests/test_api.py
from fastapi.testclient import TestClient
from app import app  # Assuming your FastAPI app is named `app`

client = TestClient(app)

def test_get_home():
    """Test the GET / endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]  # Modify based on actual response content

def test_predict_less_than_50K():
    """Test the ML model predicting '<=50K' income"""
    input_data = {
        "age": 25,
        "workclass": "Private",
        "fnlwgt": 200000,
        "education": "Some-college",
        "education_num": 10,
        "marital-status": "Never-married",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"

def test_predict_greater_than_50K():
    """Test the ML model predicting '>50K' income"""
    input_data = {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 150000,
        "education": "Masters",
        "education_num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"