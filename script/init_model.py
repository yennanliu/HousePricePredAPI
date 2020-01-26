import sys
sys.path.append(".")
from api.app import app 

def init_sample_model():
    with app.test_client()as c:
        response = c.get('/REST/api/v1.0/train')
        print ("response :", response)


if __name__ == '__main__':
    init_sample_model()