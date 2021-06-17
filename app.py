import sys
import json
from flask import Flask, request, jsonify
import keras
import pandas as pd
import numpy as np

# Environment variables
MODEL_DIR = "ml_model"
#WEIGHTS_PATH = MODEL_DIR + "/simpleModelCheckpoint.h5"

# Load Keras Model
print("Loading keras model..")
model = keras.models.load_model(
    MODEL_DIR, custom_objects=None, compile=True, options=None
)
# model.load_weigths(WEIGHTS_PATH)

# mean and std of the initial training-dataset
mean = [-1.75083294e+03,  2.00750551e+02,  1.66239649e+09,  1.19796115e+01,
        1.91216897e+00,  4.49067901e+00,  9.55689590e+00,  2.18859067e+02,
        2.18968203e+01,  4.37936406e+00,  2.73739641e+03]
std = [1.83037727e+02, 9.15228348e+01, 7.60116407e+08, 3.36948488e+00,
       1.38422981e+00, 1.60899072e+00, 2.72736533e+00, 7.63923341e+01,
       7.56692744e+00, 1.51338549e+00, 1.01327208e+03]


def normalizeTrainingData(data):
    return (data - mean) / std


valDataCSV = pd.read_csv('data/train.csv', ',')

titles = ['Lateness',
          'Assembly',
          'Material',
          'OpenOrders',
          'NewOrders',
          'TotalWork',
          'TotalSetup',
          'SumDuration',
          'SumOperations',
          'ProductionOrders',
          'CycleTime']
#
# for c in train.columns:
#     titles.append(c);

features = valDataCSV[titles]
valFeatures = normalizeTrainingData(features.values)
valFeatures = pd.DataFrame(valFeatures)
x_val = valFeatures[[i for i in range(len(titles) - 1)]].values
y_val = valFeatures.iloc[:][[len(titles)-1]].values

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
        Lateness = sended_data['Lateness']
        Assembly = sended_data['Assembly']
        Material = sended_data['Material']
        OpenOrders = sended_data['OpenOrders']
        NewOrders = sended_data['NewOrders']
        TotalWork = sended_data['TotalWork']
        TotalSetup = sended_data['TotalSetup']
        SumDuration = sended_data['SumDuration']
        SumOperations = sended_data['SumOperations']
        ProductionOrders = sended_data['ProductionOrders']
        CycleTime = 0

        # create feature numpy array
        features = np.array(
            [[[Lateness, Assembly, Material, OpenOrders, NewOrders, TotalWork, TotalSetup, SumDuration, SumOperations, ProductionOrders]]])
        # print(features)

        # normalize features
        normalized_features = normalize(features)
        # print(normalized_features)
        print("normalized", normalized_features)
        # predict cycletime with given model
        predicted_cycletime = model.predict(normalized_features)

        # denormalize
        predicted_cycletime = denormalize(predicted_cycletime)
        print(predicted_cycletime)

        # create json response object
        response_json = {'CycleTime': json.dumps(
            predicted_cycletime[0, 0].item())}
        print(response_json)

        # convert to response json object
        response = jsonify(response_json)
        response.status_code = 200
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content": exception_message})
        response.status_code = 400
    return response


@app.route('/train/cycletime.json', methods=['POST'])
def train_cycle_time():
    """
    get c# json object and predict cycletime
    """
    status = "no Training"
    historyResponse = ""
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
        CycleTime = sended_data['CycleTime']

        trainData = np.array([[Assembly, Material, TotalWork, TotalSetup,
                             SumDuration, SumOperations, ProductionOrders, CycleTime]])
        if len(trainData) == 1:
            # normalize features
            normalized_features = normalizeTrainingData(trainData)
            print("normalized_features", normalized_features)

            # create trainset
            x_train = [[normalized_features[0, 0:7]]]
            y_train = [normalized_features[0, 7]]

            print("x_train", x_train)
            print("y_train", y_train)

            # dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            #    x_train,
            #    y_train,
            #    sequence_length=1,
            #    sampling_rate=1,
            #    batch_size=1
            # )

            lr_scheduler = keras.callbacks.LearningRateScheduler(
                scheduler, verbose=1)

            modelckpt_callback = keras.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filepath=MODEL_DIR + "/simpleModelCheckpoint.h5",
                verbose=1,
                save_weights_only=True,
                save_best_only=True
            )

            # train cycletime with given model
            history = model.fit(
                x=np.array(x_train),
                y=np.array(y_train),
                #validation_data=(np.array([x_val]), np.array(y_val.values)),
                batch_size=1,
                epochs=100,
                callbacks=[modelckpt_callback, lr_scheduler])

            print(history)
            historyResponse = history
            status = "Training completed"

        # create json response object
        response_json = {
            'Learning': json.dumps(status),
            'History': json.dumps(historyResponse),
            'Length Traindata': len(trainData)}
        print(response_json)

        # convert to response json object
        response = jsonify(response_json)
        response.status_code = 200
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content": exception_message})
        response.status_code = 400
    return response


@app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        return "Working Fine"


def normalize(data):
    return (data - mean[:-1]) / std[:-1]


def denormalize(value):
    data_mean = mean[len(mean)-1]
    data_std = std[len(std)-1]
    return value*data_std+data_mean


def scheduler(epoch, lr):
    return 0.001


if __name__ == "__main__":
    # run flask application in debug mode
    app.run(debug=True, host="0.0.0.0", port=5000)
