import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics, linear_model, preprocessing
from sklearn.metrics import mean_absolute_error, r2_score
import os
import pickle

def load_data():
    df_train = pd.read_csv("data/train.csv")
    df_test  = pd.read_csv("data/test.csv")
    return df_train, df_test

def preprocess_data(df_train, df_test):   
    # Concatenate df_train and df_test data together
    data = pd.concat([df_train, df_test], sort=False)   
    # Label Encoding
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))           
    # Fill in the missing data
    data.fillna(0, inplace=True)
    df_train = data[: len(df_train)]
    df_test = data[-len(df_test) :]
    return df_train, df_test

def process_data(train, test):
    y = train["SalePrice"]
    # Drop all the ID variables
    X = train.drop(["Id", "SalePrice"], axis=1)
    X_test_ = test.drop(["Id", "SalePrice"], axis=1)
    # train, test split
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
    return X_train, X_test, y_train,  y_test, X_test_

def train_model(X_train, X_test, y_train,  y_test, X_test_):
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
    model_path = "model/regr_model.pickle"
    pickle.dump(regr, open("model/regr_model.pickle", 'wb'))
    return result

def predict(x_test):
    # load model
    model_path = "model/regr_model.pickle"
    model = pickle.load(open(model_path,'rb'))
    print ("model", model)
    y_pred = model.predict(x_test)
    return y_pred

if __name__ == '__main__':
    df_train, df_test = load_data()
    train, test = preprocess_data(df_train, df_test)
    X_train, X_test, y_train,  y_test, X_test_ = process_data(train, test)
    if os.path.isfile("model/regr_model.pickle"):
        print (">>> load pre-train model")
        result = predict(X_test_)
        print (result)
    else:
        print (">>> train model")
        result = train_model(X_train, X_test, y_train,  y_test, X_test_)
        print (result)

