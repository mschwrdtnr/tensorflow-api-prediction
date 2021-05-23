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
weightPath = "kerasModel/simpleModelCheckpoint.h5"
model.load_weights(weightPath)


split_fraction = 0.70
train_split = int(split_fraction * int(train.shape[0]))
step = 1

batch = 128
epochs = 100
mean = 0
std = 0

def normalize(data, train_split):
    global mean
    global std
    data_mean = data[:train_split].mean(axis=0)
    mean = data_mean
    data_std = data[:train_split].std(axis=0)
    std = data_std
    return (data - data_mean) / data_std

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
    
features = train[titles]
features = normalize(features.values, train_split)
features = pd.DataFrame(features)

train_data = features.loc[0 : train_split - 1] #Training Data
val_data = features.loc[train_split:] #Validation Data

#start = past + future
start = 0
end = start + train_split

x_train = train_data[[i for i in range(len(titles))]].values
y_train = features.iloc[start:end][[len(titles)-1]]

#sequence_length = int(past / step)
sequence_length = 1

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch
)

label_start = train_split
valRange = int(train.shape[0]) - train_split

x_val = val_data[[i for i in range(len(titles))]].values
y_val = features.iloc[label_start:][[len(titles)-1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch
)

for batch in dataset_train.take(1):
    inputs, targets = batch

learning_rate = 0.001
model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99), loss=keras.losses.MeanAbsoluteError(), metrics=keras.metrics.MeanSquaredError())

def scheduler(epoch, lr):
    return 0.0001
    if lr > 0.004:
        return lr - 0.0002
    else:
        if lr > 0.0004:
            return lr - 0.000001
        else:            
            return 0.0001

es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=weightPath,
    verbose=1,
    save_weights_only=True,
    save_best_only=True
)

epochs = 4000

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback,modelckpt_callback, lr_scheduler]
)

model.save(kerasModel)