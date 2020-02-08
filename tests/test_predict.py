import pytest
import os
import pandas as pd 
# REST flask app 
import sys
sys.path.append(".")
from Predict.predictor import HousePricePredictor

def test_process_data():
    """
    test the _process_data method : test if can process data properly
    """
    h = HousePricePredictor()
    df_train, df_test = h._process_data() 
    assert str(type(df_train)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(df_test)) == "<class 'pandas.core.frame.DataFrame'>"

def test_process_input_data():
    """
    test the _process_input_data method : test if can process the user input data properly
    """
    df = pd.DataFrame()
    h = HousePricePredictor()
    df_ = h._process_input_data(df)
    assert str(type(df_)) == "<class 'pandas.core.frame.DataFrame'>"

def test_prepare_train_data():
    """
    test the _prepare_train_data method : test if can prepare to-train data properly
    """
    h = HousePricePredictor()
    X_train, X_test, y_train, y_test, X_test_ = h._prepare_train_data()
    assert str(type(X_train)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(X_test)) == "<class 'pandas.core.frame.DataFrame'>"
    assert str(type(y_train)) == "<class 'pandas.core.series.Series'>"
    assert str(type(y_test)) == "<class 'pandas.core.series.Series'>"
    assert str(type(X_test_)) == "<class 'pandas.core.frame.DataFrame'>"

def test_train():
    """
    test the _train method : test if the return of train method is as expected
    """
    h = HousePricePredictor()
    result = h._train()
    assert str(type(result)) == "<class 'list'>"

def test_predict():
    """
    test the _predict method : test if the return of predict method is as expected
    """
    h = HousePricePredictor()
    result = h._predict()
    assert type(result) == list

def test_predict_with_input():
    """
    test the _predict_with_input method : test if the return of predict-with-input method is as expected
    """
    input_json = {"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}
    h = HousePricePredictor()
    result = h._predict_with_input(input_json)
    assert type(result) == float

def test_predict_with_nonvalidated_input():
    """
    test the _predict_with_input method : test if the return of predict-with-input method with non-validated is as expected
    """
    input_json = {"myjson":123}
    h = HousePricePredictor()
    result = h._predict_with_input(input_json)
    assert result == None

if __name__ == '__main__':
    pytest.main([__file__])