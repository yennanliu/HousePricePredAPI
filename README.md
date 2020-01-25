# HousePricePredAPI

## INTRO
- Develop a web service (ML api) can predict the price of house based on the [Ames housing dataset](https://www.kaggle.com/c/ames-housing-data). The web serice expose one end point that can take numerical input (79 variables json) and return prediction as output

## Architecture
- dev

## Process

```
Local dev -> Local train -> Unit-test -> Docker build -> Travis (CI/CD) -> Deploy to Dockerhub  -> Deploy to AWS ECS -> Online train -> API ready
```

## File structure 

```
├── Dockerfile        : Dockerfile build web service (ML api)
├── Predict           : Main class for ML prediction
├── api               : API runner (flask web server)
├── data              : Train, and test data
├── log               : Service log file
├── model             : File storage trained models 
├── output            : File storage ML prediction output
├── requirements.txt  : Python dependency
├── script            : Helper scripts (parse json, upload files..)
└── tests             : Unit-test scripts
```

## Tech
- Python3, Flask, scikit-learn, Pandas, Numpy, pytest
- AWS ECS, ECR
- Docker 
- Travis
- Git

## Quick start

### Quick start Docker

<details>
<summary>Quick start Docker</summary>

```bash
# Docker
$ docker build . -t house_pred_env
$ docker run -p 8000:8000 -it house_pred_env 
$ curl http://localhost:8000/REST/api/v1.0/train
$ curl -i -H "Content-Type: application/json" -X POST -d $(python script/get_test_json.py) http://localhost:8000/REST/api/v1.0/predict_with_input

```
</details>

### Quick start maunally

<details>
<summary>Quick start maunally</summary>

```bash
# Maunally method I 
$ python api/app.py
$ curl http://localhost:8000/REST/api/v1.0/train
$  curl -i -H "Content-Type: application/json" -X POST -d $(python script/get_test_json.py) http://localhost:8000/REST/api/v1.0/predict_with_input

# Maunally method II
$ python api/app.py
$ curl -i -H "Content-Type: application/json" -X POST -d '{"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}' http://localhost:8000/REST/api/v1.0/predict_with_input
```
</details>

### Useage of the API

<details>
<summary>Useage of the API</summary>

1. Check API status
Endpoint: `/`

```bash
$ curl http://localhost:8000/
# API Hello World!
```

2. Train a model
Endpoint: `/REST/api/v1.0/train`

```bash
$ curl http://localhost:8000/REST/api/v1.0/train

```

3. Predict on the test data
Endpoint: `/REST/api/v1.0/predict`

```bash
$ curl http://localhost:8000/REST/api/v1.0/predict

```

4. Predict on input json 
Endpoint: `/REST/api/v1.0/predict_with_input`

```bash 
$ curl -i -H "Content-Type: application/json" -X POST -d '{"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}' http://localhost:8000/REST/api/v1.0/predict_with_input

```

5. List trained models 
Endpoint: `/REST/api/v1.0/model_list`

```bash
$ curl http://localhost:8000/REST/api/v1.0/model_list

```

6. List ML predictions
Endpoint: `/REST/api/v1.0/predict_list`

```bash
$ curl http://localhost:8000/REST/api/v1.0/predict_list

```

</details>

## Development 

<details>
<summary>Development</summary>

```bash
# unit test 
$ pytest -v tests/
# ============================ test session starts =============================
# platform darwin -- Python 3.6.10, pytest-5.3.3, py-1.8.1, pluggy-0.13.1 -- /Users/yennanliu/anaconda3/envs/yen_dev/bin/python
# cachedir: .pytest_cache
# rootdir: /Users/yennanliu/HousePricePredAPI
# collected 18 items                                                           

# tests/test_api.py::test_404_page_not_found PASSED                      [  5%]
# tests/test_api.py::test_api_helloworld PASSED                          [ 11%]
# tests/test_api.py::test_get_model_list PASSED                          [ 16%]
# tests/test_api.py::test_get_predict_list PASSED                        [ 22%]
# tests/test_api.py::test_train_house_price_model PASSED                 [ 27%]
# tests/test_api.py::test_predict_house_price PASSED                     [ 33%]
# tests/test_api.py::test_predict_house_price_with_input PASSED          [ 38%]
# tests/test_predict.py::test_list_model PASSED                          [ 44%]
# tests/test_predict.py::test_list_prediction PASSED                     [ 50%]
# tests/test_predict.py::test_save_model PASSED                          [ 55%]
# tests/test_predict.py::test_load_model PASSED                          [ 61%]
# tests/test_predict.py::test_process_data PASSED                        [ 66%]
# tests/test_predict.py::test_process_input_data PASSED                  [ 72%]
# tests/test_predict.py::test_prepare_train_data PASSED                  [ 77%]
# tests/test_predict.py::test_train PASSED                               [ 83%]
# tests/test_predict.py::test_predict PASSED                             [ 88%]
# tests/test_predict.py::test_predict_with_input PASSED                  [ 94%]
# tests/test_predict.py::test_predict_with_nonvalidated_input PASSED     [100%]

# ============================== warnings summary ==============================

```
</details>

## Deployment 
- dev 

## TODO

<details>
<summary>TODO</summary>

- Offline training 
- Online training (when new input data, save the re-train model as new version)
- Train (via API) with super-parameter / parameter
- Output model as standard format
- Input validation
- Error handling 
- Scaling up
- Track log (?)

</details>
