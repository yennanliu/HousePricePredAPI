import requests
import pytest
# REST flask app 
import sys
sys.path.append(".")
from api.app import app 

def test_404_page_not_found():
    """
    test if can return http 404 error if input invalid url 
    """
    with app.test_client() as c:
        response = c.get('/this_page_not_exists')
        assert response.status_code == 404

def test_api_helloworld():
    """
    test if can return http 200 if calling the simplest url 
    """
    with app.test_client() as c:
        response = c.get('/')
        assert response.status_code == 200

def test_get_model_list():
    pass 

def test_get_predict_list():
    pass 

def test_train_house_price_model():
    with app.test_client()as c:
        response = c.get('/REST/api/v1.0/_train')
        print ("response :", response)
        assert response.status_code == 201 

def test_predict_house_price():
    with app.test_client()as c:
        response = c.get('/REST/api/v1.0/_predict')
        print ("response :", response)
        assert response.status_code == 201 

def test_predict_house_price_with_input():
    headers={"Content-Type": "application/json"}
    data = '{"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}'
    with app.test_client()as c:
        response = c.post('/REST/api/v1.0/_predict_with_input', headers=headers, data=data)
        assert response.status_code == 201

if __name__ == '__main__':
    pytest.main([__file__])