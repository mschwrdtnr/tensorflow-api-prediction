import sys
import json
from flask import Flask, request, jsonify
import keras
import pandas as pd
import numpy as np

# Environment variables
MODEL_DIR= "ml_model"
WEIGHTS_PATH = MODEL_DIR + "/simpleModelCheckpoint.h5"

# Load Keras Model
print("Loading keras model..")
model = keras.models.load_model(
    MODEL_DIR, custom_objects=None, compile=True, options=None
)
#model.load_weigths(WEIGHTS_PATH)

# mean and std of the initial training-dataset
mean = [8.38349267e+01, 7.14052754e+08, 5.58996266e+00, 9.59101222e+00, 2.21157340e+02, 2.19141673e+01, 4.38283347e+00, 3.19661680e+03]
std = [2.11527778e+01, 1.80308080e+08, 2.40437363e+00, 2.29533925e+00, 8.77253571e+01, 8.74016707e+00, 1.74803341e+00, 1.65355304e+03]

# Initialize flask application
app = Flask(__name__)

# Test with curl -X POST -H 'Content-Type:application/json' http://localhost:5000/predict/cycletime.json -d '{"Assembly":19.841,"Material":187106750,"TotalWork":3.9,"TotalSetup":9.03,"SumDuration":158,"SumOperations":15,"ProductionOrders":3}'
@app.route('/predict/cycletime.json', methods=['POST'])
def predict_cycle_time():
    """
    get c# json object and predict cycletime
    """
    try:                
        # get request json object
        sended_data = request.get_json()
        print(sended_data)
        # features = np.array(list(map(float, features.split(','))))

        # extract vars of send json
        Assembly = sended_data['Assembly']
        Material = sended_data['Material']
        TotalWork = sended_data['TotalWork']
        TotalSetup = sended_data['TotalSetup']
        SumDuration = sended_data['SumDuration']
        SumOperations = sended_data['SumOperations']
        ProductionOrders = sended_data['ProductionOrders']
        CycleTime = 0

        # create feature numpy array
        features = np.array([[[Assembly, Material, TotalWork, TotalSetup, SumDuration, SumOperations, ProductionOrders, CycleTime]]])
        print(features)

        # normalize features
        normalized_features = normalize(features)
        print(normalized_features)

        # predict cycletime with given model
        predicted_cycletime = model.predict(normalized_features)

        # denormalize
        predicted_cycletime = denormalize(predicted_cycletime)
        print(predicted_cycletime)

        # create json response object
        response_json = {'CycleTime':json.dumps(predicted_cycletime[0,0,0].item())}
        print(response_json)

        # convert to response json object 
        response = jsonify(response_json)
        response.status_code = 200  
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content":exception_message})
        response.status_code = 400
    return response

@app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        return "Working Fine"

def normalize(data):
    return (data - mean) / std

def denormalize(value):
    data_mean = mean[len(mean)-1]
    data_std = std[len(std)-1]
    return value*data_std+data_mean

if __name__ == "__main__":
    # run flask application in debug mode
    app.run(debug=True)
