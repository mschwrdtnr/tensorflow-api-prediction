#!/usr/bin/python
import numpy as np
import pandas as pd
import keras
import math

# mean and std of the initial training-dataset
mean = [8.38349267e+01, 7.14052754e+08, 5.58996266e+00, 9.59101222e+00, 2.21157340e+02, 2.19141673e+01, 4.38283347e+00, 3.19661680e+03]
std = [2.11527778e+01, 1.80308080e+08, 2.40437363e+00, 2.29533925e+00, 8.77253571e+01, 8.74016707e+00, 1.74803341e+00, 1.65355304e+03]
# mean = pd.read_csv('config/mean.csv')
# std = pd.read_csv('config/std.csv')

# train = pd.read_csv('TestData/full_trainset.csv',',')
modelPath = "kerasModel"
model = keras.models.load_model(modelPath)
model.load_weights(modelPath + "/simpleModelCheckpoint.h5")

def normalize(data):
    return (data - mean) / std

def denormalize(value):
    data_mean = mean[len(mean)-1]
    data_std = std[len(std)-1]
    return value*data_std+data_mean

titles = ['Assembly',
          'Material',
#           'OpenOrders',
#           'NewOrders',
          'TotalWork',
          'TotalSetup',
          'SumDuration',
          'SumOperations',
          'ProductionOrders',
          'CycleTime']

# Prediction Test
predTest = pd.read_csv('TestData/testData1.csv',';')
featuresPredTest = predTest[titles]
predTestRange = int(featuresPredTest.shape[0])
featuresPredTest = normalize(featuresPredTest.values)
featuresPredTest = pd.DataFrame(featuresPredTest)
x_predTest = featuresPredTest[[i for i in range(len(titles))]].values
y_predTest = featuresPredTest.iloc[0:][[len(titles)-1]]

datasetPredTest = keras.preprocessing.timeseries_dataset_from_array(
    x_predTest,
    y_predTest,
    sequence_length=1,
    sampling_rate=1,
    batch_size=1
)

for x, y in datasetPredTest.take(1):
    predictionData = model.predict(x)
    denormalized_predictionData = denormalize(predictionData)
    
print(denormalized_predictionData[0][0][0])