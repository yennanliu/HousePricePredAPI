import pytest
import os
import pandas as pd 
# REST flask app 
import sys
sys.path.append(".")
from utils.file_io import FileIO

def test_list_model():
    """
    test the _list_model method : list trained models
    """
    f = FileIO()
    models = f._list_model()
    assert type(models) == dict

def test_list_prediction():
    """
    test the _list_prediction method : list of ML predictions
    """
    f = FileIO()
    predictions = f._list_prediction()
    assert type(predictions) == str

def test_save_model():
    """
    test the _save_model method : test if can save model properly
    """
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    f = FileIO()
    result = f._save_model(model, "dummy model evaluation")
    assert result == True

def test_load_model():
    """
    test the _load_model method : test if can load model properly
    """
    f = FileIO()
    model = f._load_model()
    assert str(type(model)) == "<class 'sklearn.linear_model._base.LinearRegression'>"  or str(type(model)) == "<class 'sklearn.linear_model.base.LinearRegression'>" 
