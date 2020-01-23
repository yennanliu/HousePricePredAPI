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

if __name__ == '__main__':
    pytest.main([__file__])