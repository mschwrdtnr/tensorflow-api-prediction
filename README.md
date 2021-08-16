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

## Explanation of the python code
@MaxWeickert

## How to change something for own purpose
- Use other variables for prediction?
@MaxWeickert

## Train a new Model
@MaxWeickert

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
