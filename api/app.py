import os, json 
from flask import Flask, jsonify, make_response, request, render_template
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
    return make_response(jsonify({'error': 'HTTP 404 : Not found'}), 404)

# return 400 if bad request
@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'HTTP 400 : bad request'}), 404)

@app.route('/REST/api/v1.0/doc')
def get_doc():
    return render_template('index.html') 

@app.route('/REST/api/v1.0/model_list')
def get_model_list():
    cli_args = request.json 
    h = HousePricePredictor()
    model_list = h._list_model()
    print (model_list)
    return {'model_list': model_list}, 201

@app.route('/REST/api/v1.0/predict_list')
def get_predict_list():
    cli_args = request.json 
    h = HousePricePredictor()
    prediction_list = h._list_prediction()
    print (prediction_list)
    return {'prediction_list': prediction_list}, 201

# call house price prediction model
@app.route('/REST/api/v1.0/train')
def train_house_price_model():
    cli_args = request.json 
    h = HousePricePredictor()
    response = h._train()
    #print (response)
    return {'train_result': response}, 201

@app.route('/REST/api/v1.0/predict')
def predict_house_price():
    cli_args = request.json 
    h = HousePricePredictor()
    response = h._predict()
    #print (response)
    if not response:
        return {'pred_result': None}, 400
    return {'pred_result': response}, 201

@app.route('/REST/api/v1.0/predict_with_input', methods=['POST'])
def predict_house_price_with_input():
    cli_args = request.json 
    h = HousePricePredictor()
    input_data = cli_args
    response = h._predict_with_input(input_data)
    if not response:
        return {'train_result': None}, 400
    return {'train_result': response}, 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)