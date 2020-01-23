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

def test_load_model():
    h = HousePricePredictor()
    model = h._load_model()
    assert str(type(model)) == "<class 'sklearn.linear_model._base.LinearRegression'>"

def test_process_data():
    h = HousePricePredictor()
    df_train, df_test = h._process_data() 
    assert str(type(df_train)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(df_test)) == "<class 'pandas.core.frame.DataFrame'>"

if __name__ == '__main__':
    pytest.main([__file__])