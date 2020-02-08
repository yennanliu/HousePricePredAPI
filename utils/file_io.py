import os, pickle, json
from datetime import datetime

class FileIO:
    """
    class load/dump data from local file system
    """
    def _list_model(self):
        """
        list saved models
        """
        def load_json_file(filename):
            with open(filename, 'r') as f:
                data=f.read()
            return json.loads(data)

        models = [ m for m in os.listdir("model") if m.endswith(".pickle")]
        models_evals = [ m for m in  os.listdir("model_eval")]
        model_details = {}
        for m in models:
            model_eval_file= "model_eval/" + m.split(".")[0] + '_eval.json'
            model_json = load_json_file(model_eval_file)
            model_details[m] = model_json
        return  model_details

    def _list_prediction(self):
        """
        list ML prediction outputs
        """
        predictions = os.listdir("output")
        return str(predictions)

    def _save_model(self, model, eval_metric):
        """
        method to save model as pickle
        : input  : sklearn model object 
        : output : python pickle file 
        """
        now = datetime.now()
        current_time = now.strftime('%Y-%m-%d-%H:%M:%S')
        try:
            model_name = "model/model_{}.pickle".format(current_time)
            pickle.dump(model, open(model_name, 'wb'))
            print (">>> Save model OK : ", model_name)
            model_eval = "model_eval/model_{}_eval.json".format(current_time)
            with open(model_eval, 'w') as f:
                json.dump(eval_metric, f)
                print (">>> Save model eval OK : ", model_eval)
            return True
        except Exception as e:
            print (">>> Model save failed", str(e))
            return False

    def _save_output(self, df):
        """
        method to save output as csv
        : input  : pandas dataframe 
        : output : csv file
        """
        now = datetime.now()
        current_time = now.strftime('%Y-%m-%d-%H:%M:%S')
        try:
            output_name  = "output/pred_output_{}.csv".format(current_time)
            df.to_csv(output_name, index=False)
            print (">>> Save output OK : ", output_name)
        except Exception as e:
            print (">>> output save failed", str(e))
            return False

    def _load_model(self, model=None):
        """
        method to load saved model, if no input model name, will load the "latest" model (timestamp)
        : input  : model name (string)
        : output : sklearn model object 
        """
        models = os.listdir("model")
        # if no any saved model, then have to train one first 
        if not models:
            print (">>> No saved model, please train fitst")
            return 
        # load the model fit given name
        elif model != None:
            return models[model]
        model_dict = dict()
        for model in models:
            model_dict[model.split("_")[1].split(".")[0]] = model
        # load the latest model (timestamp)
        max_model_idx = max(model_dict.keys())
        model_name = model_dict[max_model_idx]
        return pickle.load(open("model/" + model_name,'rb'))
