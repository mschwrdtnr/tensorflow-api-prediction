# tensorflow-api-prediction
This project is part of a research and development seminar at HTW Dresden with the goal to test Machine Learning in an [self-organizing production](https://github.com/Krockema/MATE).

With this project you can setup a container launching a REST API to call prediction in the self organizing production based on input data. It's also possible to train a new keras model with the given implementation.

This repository is structured as follows:
```md
tensorflow-rest-prediction
| README.md
| app.py                      # Launch file of the python programming 
| docker-compose.yml          # Descriptions for launching the docker container
| Dockerfile                  # Descriptions for launching the docker container
| CSharpPythonRESTfulAPI.cs   # File for REST Message to predict with container
| requirements.txt            # Python dependencies which are automated installed while building the container
|-- Data                      # Folder for train and validation data
|-- python_scripts            # Old python scripts for training and prediction
|-- Notebooks	              # Old notebooks for training and prediction
|-- ml_model                  # Place of all necessary data of the trained model
```

## Usage
### Docker Compose
1. `docker-compose build`
2. `docker-compose up`

> Make sure that a current docker and docker-compose version is installed.
### Python Script
1. `pip install -r requirements.txt`
2. `python app.py`

> Make sure that Python 3 is installed.

## Explanation of the prediction part of the api
The method for the prediction consists out of 5 main parts:

1. Load the mdoel
For that you'll need to first access the directory of your machine learning model. Therefore you generate two variables:
```
MODEL_DIR = "ml_model"
WEIGHTS_PATH = MODEL_DIR + "/simpleModelCheckpoint.h5"
```
To load the model you'll need to execute the following lines of code:
```
model = keras.models.load_model(
    MODEL_DIR, custom_objects=None, compile=True, options=None
)
model.load_weights(WEIGHTS_PATH)
```
2. Tell the API which variables you expect
First access the received data and feed them into variables and the variables into a numpy array:
```
sended_data = request.get_json()
# example variables
Lateness = sended_data['Lateness']
OpenOrders = sended_data['OpenOrders']
features = np.array([[[Lateness, OpenOrders]]])
```
3. Normalize the data if your model expects normalized data
You can use a function which uses the mean and standard of all your input parameters of your model taken from the training data. Exclude your target variable (last entry of the array):
```
def normalize(data):
    return (data - mean[:-1]) / std[:-1]
```
Call the function:
```
normalized_features = normalize(features)
```
4. Predict the target variable
```
predicted_cycletime = model.predict(normalized_features)
```
5. Denormalize the predicted value
Use the denormalize function with the last entry of the same mean and standard array as for the normalization
```
def denormalize(value):
    data_mean = mean[len(mean)-1]
    data_std = std[len(std)-1]
    return value*data_std+data_mean
```
Call the function:
```
predicted_cycletime = denormalize(predicted_cycletime)
```
6. Build a json object and send it back to the client
```
response_json = {'CycleTime': json.dumps(predicted_cycletime[0, 0].item())}
response = jsonify(response_json)
response.status_code = 200

return response
```

## Train a new Model
If your goal is to train a new Model you can use the Training.ipynb file out of the folder which is called "Notebooks". It's a jupyter notebook for exactly that purpose.
These are the steps you may take if you want to adjust the notebook for your own use:

1. Choose the right .csv file with your training data
```
train = pd.read_csv('../data/train.csv',',')
```
2. Edit the titles to equal your csv headers or just the columns you want to use
```
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
```
3. Change up the parameters of your neural network like the number of layers and neurons, type of layers, optimizer and metrics
```
inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
dense1 = keras.layers.Dense(512, activation="tanh")(inputs)
dense2 = keras.layers.Dense(512, activation="tanh")(dense1)
dense3 = keras.layers.Dense(512, activation="tanh")(dense2)
dense4 = keras.layers.Dense(512, activation="tanh")(dense3)
outputs = keras.layers.Dense(1)(dense4)
learning_rate = 0.001
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99), loss=keras.losses.MeanSquaredError(), metrics=keras.metrics.MeanAbsoluteError())
```
For further information you may consolidate the [Keras API](https://keras.io/api/)

4. Start the training with your chosen number of epochs
```
epochs = 4000

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback,modelckpt_callback, lr_scheduler]
)
```
5. Visualize the training and validation loss to see how your model performs
```
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
visualize_loss(history, "Training and Validation loss")
```

## Testing
### Test the prediction
You can test the api with curl as follows:
```
curl -X POST -H 'Content-Type:application/json' http://localhost:5000/predict/cycletime.json -d '{"Lateness":-1690,"Assembly":19.841,"Material":187106750,"OpenOrders":11,"NewOrders":1,"TotalWork":3.9,"TotalSetup":9.03,"SumDuration":158,"SumOperations":15,"ProductionOrders":3}'
```

You should get the following answer:
```
{
  "CycleTime": "1936.75146484375"
}
```
### Test the status of the prediction
`curl http://localhost:5000/health`


## Used Tutorials
We used a couple of tutorials to make this work:
- https://ernest-bonat.medium.com/using-c-to-call-python-restful-api-web-services-with-machine-learning-models-6d1af4b7787e
- https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
- https://medium.com/geekculture/machine-learning-prediction-in-real-time-using-docker-python-rest-apis-with-flask-and-kubernetes-fae08cd42e67
- https://towardsdatascience.com/serving-a-machine-learning-model-via-rest-api-5a4b38c02e90
- https://hacheemaster.github.io/DeployRESTAPIDocker/#
- https://github.com/deepakiim/Deploy-machine-learning-model
