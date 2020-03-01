import os, json 
from flask import Flask, jsonify, make_response, request, render_template
import numpy as np
# UDF 
import sys
sys.path.append(".")
from Predict.predictor import HousePricePredictor
from utils.file_io import FileIO

app = Flask(__name__)

h = HousePricePredictor()
f = FileIO()

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

# html page for API doc
@app.route('/REST/api/v1.0/doc')
def get_doc():
    return render_template('doc.html') 

# check flask server (api service) health
@app.route('/REST/api/v1.0/health')
def get_service_status():
    with app.test_client() as client:
        response = client.get('/')
        if response.status_code==200:
            api_status = 'OK'
        else:
            api_status = 'Error'
    return jsonify(api_status=api_status, http_status=response.status_code)

# return list of models and their performance
@app.route('/REST/api/v1.0/model_list')
def get_model_list():
    cli_args = request.json 
    model_list = f._list_model()
    print (model_list)
    return jsonify(model_list=model_list), 201

# return list of house price predictions 
@app.route('/REST/api/v1.0/predict_list')
def get_predict_list():
    cli_args = request.json 
    prediction_list = f._list_prediction()
    print (prediction_list)
    return jsonify(prediction_list=prediction_list), 201

# train the house price prediction model
@app.route('/REST/api/v1.0/train')
def train_house_price_model():
    cli_args = request.json 
    response = h._train()
    #print (response)
    return jsonify(result=response), 201

# predict the test dataset with house price prediction model
@app.route('/REST/api/v1.0/predict')
def predict_house_price():
    cli_args = request.json 
    response = h._predict()
    #print (response)
    if not response:
        return jsonify(result=None), 400
    return jsonify(result=response), 201

# predict the house price with JSON input
@app.route('/REST/api/v1.0/predict_with_input', methods=['POST'])
def predict_house_price_with_input():
    cli_args = request.json 
    input_data = cli_args
    response = h._predict_with_input(input_data)
    #if not response:
    #    return jsonify(train_result=None), 400
    return jsonify(result=response), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)