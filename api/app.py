import os, json 
from flask import Flask, jsonify, make_response, request
import numpy as np
# UDF 
import sys
sys.path.append(".")
from Predict.predictor import HousePricePredictor

app = Flask(__name__)

# API hello world 
@app.route('/')
def index():
    return "API Hello World!", 200

# return 404 if not valid API url 
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/REST/api/v1.0/model_list')
def get_model_list():
    cli_args = request.json 
    h = HousePricePredictor()
    model_list = h._list_model()
    print (model_list)
    return 'model list  : {}'.format(model_list), 201

@app.route('/REST/api/v1.0/predict_list')
def get_predict_list():
    cli_args = request.json 
    h = HousePricePredictor()
    prediction_list = h._list_prediction()
    print (prediction_list)
    return 'prediction list  : {}'.format(prediction_list), 201

# call house price prediction model
@app.route('/REST/api/v1.0/train')
def train_house_price_model():
    cli_args = request.json 
    h = HousePricePredictor()
    response = h._train()
    print (response)
    return '_train Result : ', 201

@app.route('/REST/api/v1.0/predict')
def predict_house_price():
    cli_args = request.json 
    h = HousePricePredictor()
    response = h._predict()
    print (response)
    if not response:
        return str(None), 400
    return '_predict Result : {}'.format(response), 201

@app.route('/REST/api/v1.0/predict_with_input', methods=['POST'])
def predict_house_price_with_input():
    cli_args = request.json 
    h = HousePricePredictor()
    print (">>> cli_args")
    print (cli_args)
    input_data = cli_args
    response = h._predict_with_input(input_data)
    if not response:
        return str(None), 400
    return (response), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)