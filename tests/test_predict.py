import requests
import pytest
import os 
# REST flask app 
import sys
sys.path.append(".")
from Predict.predictor import HousePricePredictor

def test_list_model():
    h = HousePricePredictor()
    models = h._list_model()
    assert type(models) == str

def test_list_prediction():
    h = HousePricePredictor()
    predictions = h._list_prediction()
    assert type(predictions) == str

def test_save_model():
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    h = HousePricePredictor()
    result = h._save_model(model)
    assert result == True

if __name__ == '__main__':
    pytest.main([__file__])