import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics, linear_model, preprocessing
from sklearn.metrics import mean_absolute_error, r2_score
import os, pickle, json
from pandas.io.json import json_normalize
from datetime import datetime


class HousePricePredictor:

    def __init__(self):
        self.df_train = pd.read_csv("data/train.csv")
        self.df_test  = pd.read_csv("data/test.csv")

    def _list_model(self):
        models = os.listdir("model")
        return str(models)

    def _list_prediction(self):
        predictions = os.listdir("output")
        return str(predictions)

    def _save_model(self, model):
        now = datetime.now()
        current_time = now.strftime('%Y-%m-%d-%H:%M:%S')
        try:
            model_name = "model/regr_model_{}.pickle".format(current_time)
            pickle.dump(model, open(model_name, 'wb'))
            print (">>> Save model OK : ", model_name)
        except Exception as e:
            print (">>> Model save failed", str(e))

    def _load_model(self, model=None):
        models = os.listdir("model")
        if not models:
            print (">>> No saved model, please train fitst")
            return 
        elif model != None:
            return models[model]
        model_dict = dict()
        for model in models:
            model_dict[model.split("_")[2].split(".")[0]] = model
        max_model_idx = max(model_dict.keys())
        model_name = model_dict[max_model_idx]
        return pickle.load(open("model/" + model_name,'rb'))

    def _process_data(self):
        df_train = self.df_train
        df_test = self.df_test
        data = pd.concat([df_train, df_test])   
        # Label Encoding
        for f in data.columns:
            if data[f].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                #lbl.fit(list(data[f].values))
                #data[f] = lbl.transform(list(data[f].values)) 
                del data[f]          
        # Fill in the missing data
        data.fillna(0, inplace=True)
        df_train_ = data[: len(df_train)]
        df_test_ = data[-len(df_test) :]
        return df_train_, df_test_

    def _process_input_data(self, df):
        data = df
        # Label Encoding
        for f in data.columns:
            if data[f].dtype=='object':
                #lbl = preprocessing.LabelEncoder()
                #lbl.fit(list(data[f].values))
                #data[f] = lbl.transform(list(data[f].values))     
                del data[f]      
        # Fill in the missing data
        data.fillna(0, inplace=True)
        df_ = data[: len(df)]
        return df_

    def _prepare_train_data(self):
        train, test = self._process_data()
        y = train["SalePrice"]
        # Drop all the ID variables
        X = train.drop(["Id", "SalePrice"], axis=1)
        X_test_ = test.drop(["Id", "SalePrice"], axis=1)
        # train, test split
        X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
        return X_train, X_test, y_train, y_test, X_test_

    def _train(self):
        train, test = self._process_data()
        X_train, X_test, y_train,  y_test, X_test_ = self._prepare_train_data()
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred_val = regr.predict(X_test)
        y_pred = regr.predict(X_test_)
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print('Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred_val))
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred_val))
        result = pd.DataFrame({"Id": test["Id"],"SalePrice": y_pred,})
        result.to_csv("output/submission_reg.csv", index=False)
        print ("result :", result)
        # save model
        # model_path = "model/regr_model.pickle"
        # pickle.dump(regr, open("model/regr_model.pickle", 'wb'))
        self._save_model(regr)
        return result

    def _predict(self):
        X_train, X_test, y_train,  y_test, X_test_ = self._prepare_train_data()
        # load model
        model = self._load_model()
        print ("model", model)
        try:
            y_pred = model.predict(X_test)
            return list(y_pred)
        except Exception as e:
            print (">>> Failed : predict ", str(e))
            return None

    def _predict_with_input(self, input_json):
        print ("input_json : ", input_json)
        #input_dict = json.loads(str(input_json))
        input_df = json_normalize(input_json)
        # load model
        model = self._load_model()
        print ("model", model)
        input_df_ = self._process_input_data(input_df)
        try:
            y_pred = model.predict(input_df_)
            print ("y_pred :", y_pred)
            return dict(y_pred)
        except Exception as e:
            print (">>> Failed : predict_with_input ", str(e))
            return None