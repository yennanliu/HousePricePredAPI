import pytest
import os
import pandas as pd 
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
    assert str(type(model)) == "<class 'sklearn.linear_model._base.LinearRegression'>"  or str(type(model)) == "<class 'sklearn.linear_model.base.LinearRegression'>" 

def test_process_data():
    h = HousePricePredictor()
    df_train, df_test = h._process_data() 
    assert str(type(df_train)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(df_test)) == "<class 'pandas.core.frame.DataFrame'>"

def test_process_input_data():
    df = pd.DataFrame()
    h = HousePricePredictor()
    df_ = h._process_input_data(df)
    assert str(type(df_)) == "<class 'pandas.core.frame.DataFrame'>"

def test_prepare_train_data():
    h = HousePricePredictor()
    X_train, X_test, y_train, y_test, X_test_ = h._prepare_train_data()
    assert str(type(X_train)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(X_test)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(y_train)) == "<class 'pandas.core.series.Series'>"
    assert str(type(y_test)) == "<class 'pandas.core.series.Series'>"
    assert str(type(X_test_)) == "<class 'pandas.core.frame.DataFrame'>"

def test_train():
    h = HousePricePredictor()
    result = h._train()
    assert str(type(result)) == "<class 'list'>"

def test_predict():
    h = HousePricePredictor()
    result = h._predict()
    assert type(result) == list

def test_predict_with_input():
    input_json = {"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}
    h = HousePricePredictor()
    result = h._predict_with_input(input_json)
    assert type(result) == float

def test_predict_with_nonvalidated_input():
    input_json = {"myjson":123}
    h = HousePricePredictor()
    result = h._predict_with_input(input_json)
    assert result == None

if __name__ == '__main__':
    pytest.main([__file__])