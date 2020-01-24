# HousePricePredAPI

## INTRO

## Quick start

### Quick start Docker

```bash
# Docker
$ docker build . -t env
$ docker run -p 8000:8000 -it env 
$ curl -i -H "Content-Type: application/json" -X POST -d $(python script/get_test_json.py) http://localhost:8000/REST/api/v1.0/_predict_with_input

```
### Quick start Maunally

```bash
# Maunally method I 
$ python api/app.py
$  curl -i -H "Content-Type: application/json" -X POST -d $(python script/get_test_json.py) http://localhost:8000/REST/api/v1.0/_predict_with_input

# Maunally method II
$ python api/app.py
$ curl -i -H "Content-Type: application/json" -X POST -d '{"MSSubClass":20.0,"LotFrontage":100.0,"LotArea":17500.0,"OverallQual":7.0,"OverallCond":8.0,"YearBuilt":1959.0,"YearRemodAdd":2002.0,"MasVnrArea":0.0,"BsmtFinSF1":1406.0,"BsmtFinSF2":0.0,"BsmtUnfSF":496.0,"TotalBsmtSF":1902.0,"1stFlrSF":1902.0,"2ndFlrSF":0.0,"LowQualFinSF":0.0,"GrLivArea":1902.0,"BsmtFullBath":1.0,"BsmtHalfBath":0.0,"FullBath":2.0,"HalfBath":0.0,"BedroomAbvGr":3.0,"KitchenAbvGr":1.0,"TotRmsAbvGrd":7.0,"Fireplaces":2.0,"GarageYrBlt":1959.0,"GarageCars":2.0,"GarageArea":567.0,"WoodDeckSF":0.0,"OpenPorchSF":207.0,"EnclosedPorch":162.0,"3SsnPorch":0.0,"ScreenPorch":0.0,"PoolArea":0.0,"MiscVal":0.0,"MoSold":5.0,"YrSold":2010.0}' http://localhost:8000/REST/api/v1.0/_predict_with_input

```

## Architecture

## Process
```
Local dev -> Local train -> Docker build ->  Unit-test -> Docker deployment -> Docker container to Lambda layer -> Lambda deployment (AWS) -> Online train -> Call API

```

## File structure 

## Tech

## Development 
```bash
# unit test 
$ pytest -v tests/
# ============================ test session starts =============================
# platform darwin -- Python 3.6.10, pytest-5.3.3, py-1.8.1, pluggy-0.13.1 -- /Users/yennanliu/anaconda3/envs/yen_dev/bin/python
# cachedir: .pytest_cache
# rootdir: /Users/yennanliu/HousePricePredAPI
# collected 10 items                                                           

# tests/test_api.py::test_404_page_not_found PASSED                      [ 10%]
# tests/test_api.py::test_api_helloworld PASSED                          [ 20%]
# tests/test_api.py::test_get_model_list PASSED                          [ 30%]
# tests/test_api.py::test_get_predict_list PASSED                        [ 40%]
# tests/test_api.py::test_train_house_price_model PASSED                 [ 50%]
# tests/test_api.py::test_predict_house_price PASSED                     [ 60%]
# tests/test_api.py::test_predict_house_price_with_input PASSED          [ 70%]
# tests/test_predict.py::test_list_model PASSED                          [ 80%]
# tests/test_predict.py::test_list_prediction PASSED                     [ 90%]
# tests/test_predict.py::test_save_model PASSED                          [100%]

# ============================== warnings summary ==============================

# CI/CD
```

## Deployment 
```bash 
# dev
```
## TODO

- Unit-test
- Offline training 
- Online training (when new input data, save the re-train model as new version)
- List of model / results
- Train (via API) with super-parameter / parameter
- Output model as standard format
- Input validation
- Error handling 
- Scaling up
- Track log (?)
