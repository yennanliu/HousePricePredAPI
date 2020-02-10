import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
import os, pickle, json
from pandas.io.json import json_normalize
from datetime import datetime

# UDF 
import sys
sys.path.append(".")
from utils.file_io import FileIO
from utils.s3_file_io import S3FileIO

class HousePricePredictor:
    """
    main class do house price prediction 
    """

    def __init__(self, prod_env=False):
        self.df_train = pd.read_csv("data/train.csv")
        self.df_test  = pd.read_csv("data/test.csv")
        self.df_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

    def _evaluate(self, y_true, y_pred):
        """
        method get model metrics
        : input  : y_true, y_pred : numpy array
        : output : python dict
        """
        _metrics = {
            'rmse': np.sqrt(((y_pred - y_true) ** 2).mean()),
            'explained_variance_score': metrics.explained_variance_score(y_true, y_pred),
            'mean_absolute_error': metrics.mean_absolute_error(y_true, y_pred),
            'mean_squared_error': metrics.mean_squared_error(y_true, y_pred),
            'mean_squared_log_error': metrics.mean_squared_log_error(y_true, y_pred),
            'median_absolute_error': metrics.median_absolute_error(y_true, y_pred),
            'r2_score': metrics.r2_score(y_true, y_pred)
            }
        return _metrics 

    def _process_data(self):
        """
        method process data for model train 
        : output : df_train_, df_test_ : pandas dataframe
        """
        df_train = self.df_train
        df_test = self.df_test
        data = pd.concat([df_train, df_test])   
        # remove all non-numerical columns 
        for f in data.columns:
            if data[f].dtype=='object':
                del data[f]          
        # fill in the missing data
        data.fillna(0, inplace=True)
        df_train_ = data[: len(df_train)]
        df_test_ = data[-len(df_test) :]
        return df_train_, df_test_

    def _process_input_data(self, df):
        """
        method process data for prediction (from API call)
        : input :  df  : pandas dataframe 
        : output : df_ : pandas dataframe 
        """
        #data = df[self.df_columns]
        data = df
        # remove all non-numerical columns 
        for column in data.columns:
            if data[column].dtype == 'object':     
                del data[column]      
        # fill in the missing data
        data.fillna(0, inplace=True)
        df_ = data[: len(df)]
        df_ = self._check_input_data(df_)
        return df_

    def _check_input_data(self, df):
        """
        method check/validate data from API call 
        : input :  df  : pandas dataframe 
        : output : df_ : pandas dataframe 
        """
        # check input df column type
        if list(df.columns) != list(self.df_columns):
            print (">>> input json not in the validated form, the desired form : {}".format(self.df_columns))
            return pd.DataFrame()
        for col in df.columns:
            if df[col].dtype not in ['int64', 'float64']:
                print (">>> input json type is non-validated")
                return pd.DataFrame()
        df_train_, df_test_ = self._process_data()
        for col in df.columns:
            median = df_train_[col].median()
            std = df_train_[col].std()
            left_boundary = median - 3*std
            right_boundary = median + 3*std
            column_val = float(df[col].iloc[0])
            if (column_val < left_boundary) or (column_val > right_boundary):
               print (">>> input json value is out of 3 standard deviation")
               print ("column : {}, value : {}, left_boundary : {}, right_boundary : {}".format(col, column_val, left_boundary, right_boundary))
               return pd.DataFrame()
        return df

    def _prepare_train_data(self):
        """
        method prepare X, y  data from model train
        : output :X_train, X_test, y_train,  y_test, test_  : pandas dataframe, numpy array, pandas dataframe, numpy array, numpy array 
        """
        train, test = self._process_data()
        y = train["SalePrice"]
        # Drop all the ID variables
        X = train.drop(["Id", "SalePrice"], axis=1)
        test_ = test.drop(["Id", "SalePrice"], axis=1)
        # train, test split
        X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
        return X_train, X_test, y_train, y_test, test_

    def _train(self, prod_env=False):
        """
        mothod for model train 
        : output : csv file
        """
        train, test = self._process_data()
        X_train, X_test, y_train,  y_test, test_ = self._prepare_train_data()
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        y_pred_testset = regr.predict(test_)
        # The coefficients
        #print('Coefficients: \n', regr.coef_)
        print (" >>> evaluate metric ")
        # model eval metric 
        eval_metric = self._evaluate(y_test, y_pred)
        print (eval_metric)
        # save pred output 
        result = pd.DataFrame({"Id": test["Id"],"SalePrice": y_pred_testset})
        f = FileIO()
        f._save_output(result)
        # save model
        f._save_model(regr, eval_metric)
        if prod_env == True:
            result.to_csv("result.csv")
            pickle.dump(regr, open("model.pickle", 'wb'))
            s3f = S3FileIO()
            s3f._upload_s3_file("result.csv", "output/result.csv")
            s3f._upload_s3_file("model.pickle", "model/model.pickle")
        return list(result['SalePrice'])

    def _predict(self, prod_env=False):
        """
        mothod for model predict 
        : output : python list
        """
        X_train, X_test, y_train,  y_test, test_ = self._prepare_train_data()
        # load model
        if prod_env == True:
            s3f = S3FileIO()
            s3f._download_s3_file("model/model.pickle", "model/model.pickle")
            model = pickle.load(open("model/model.pickle",'rb'))
        else:
            f = FileIO()
            model = f._load_model()
        print ("model", model)
        try:
            y_pred = model.predict(X_test)
            print (" >>> evaluate metric ")
            eval_metric = self._evaluate(y_test, y_pred)
            print (eval_metric)
            return list(y_pred)
        except Exception as e:
            print (">>> Failed : predict ", str(e))
            return None

    def _predict_with_input(self, input_json, prod_env=False):
        """
        mothod for model predict from API call
        : input  : json
        : output : python string
        """
        print ("input_json : ", input_json)
        if type(input_json) != dict:
            print (">>> input_json not in the desired form : dict") 
            return None     
        input_df = json_normalize(input_json)
        # load model
        if prod_env == True:
            s3f = S3FileIO()
            s3f._download_s3_file("model/model.pickle", "model/model.pickle")
            model = pickle.load(open("model/model.pickle",'rb'))
        else:
            f = FileIO()
            model = f._load_model()
        print ("model", model)
        input_df_ = self._process_input_data(input_df)
        try:
            y_pred = model.predict(input_df_)
            print ("y_pred :", y_pred)
            return float(y_pred[0])
        except Exception as e:
            print (">>> Failed : predict_with_input ", str(e))
            return None