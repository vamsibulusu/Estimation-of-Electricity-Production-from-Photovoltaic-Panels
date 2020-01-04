#importing the required libraries
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#reading the input csv file containing the data
df = pd.read_csv("PATH/TO/THE/INPUT/FILE")
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df['year'] = df['TimeStamp'].dt.year


#Selecting a timeframe to split the data into training and testing sets.
dtest = df.loc[df.year == 2014].copy()
dtrain = df.loc[df.year < 2014].copy()
dtrain['date'] = dtrain['TimeStamp'].dt.date
dtest['date'] = dtest['TimeStamp'].dt.date
dtrain = dtrain[dtrain.year >= 2013]
dtrain = dtrain.reset_index()
dtest = dtest.reset_index()

#Scaling of the data is necessary for quick convergence of neural network weights
scaler = MinMaxScaler(feature_range=(0, 1))
dtrain[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year']] = scaler.fit_transform(dtrain[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year']])
dtest[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year']] = scaler.transform(dtest[['sunHour', 'uvIndex.1', 'FeelsLikeC', 'HeatIndexC', 'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility', 'day_of_year']])
irrscaler = MinMaxScaler(feature_range=(0, 1))
#It is necessary to scale the input first and use the same scaler on the outputs to prevent data leakage
dtrain[['Irr']] = irrscaler.fit_transform(dtrain[['Irr']])
dtest[['Irr']] = irrscaler.transform(dtest[['Irr']])

#Seperating features required for the first part of the model involving the prediction of Irradiance
trainirr = dtrain[['uvIndex.1', 'cloudcover', 'humidity', 'tempC', 'visibility', 'day_of_year','hour_of_day']].copy()
train_out_irr = dtrain['Irr'].copy()
test_irr = dtest[['uvIndex.1', 'cloudcover', 'humidity', 'tempC', 'visibility', 'day_of_year','hour_of_day']].copy()
test_out_irr = dtest['Irr'].copy()
import joblib
from multiprocessing import Pool
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D, LeakyReLU
from keras.layers import LSTM, Flatten
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

#Definition of the Irradiance prediction model where the inputs are weather and the output is Irradiance
#The activations and number of layers have been optimized for performance and accuracy. They can be modified as per need.
def baseline_model():
    model = Sequential()
    model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


#Early stopping is used to prevent the overfitting of the model to input noise leading to a decrease in test accuracy despite increases in training accuracy
from keras.callbacks import EarlyStopping
model = 0
err = 1000000
#Due to the randomness in the initialization of the weights of a neural network, the optimization might get stuck in a local minima
#By taking the best of 5 random iterations, there is a higher probability that the global minima is captured. The number of iterations can be increased if the model is found to get easily stuck at local minima
for i in range(0,5):
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, min_delta=0.0001, verbose=1,restore_best_weights=True)
    model_iteration = KerasRegressor(build_fn=baseline_model, epochs=10000, batch_size=32, verbose=1, callbacks=[es])
    history = model_iteration.fit(trainirr,train_out_irr,validation_data = (test_irr,test_out_irr))
    if(min(history.history['val_loss']) < err):
        model = model_iteration
        err = min(history.history['val_loss'])
    del model_iteration
    del es

#The input data now consists of the input weather data and the predicted irradiance values from the neural network
train_irr = model.predict(trainirr)
test_irr = model.predict(test_irr)
dtrain['Irrcal'] = train_irr
dtest['Irrcal'] = test_irr


from scipy.spatial.distance import euclidean
from multiprocessing import Pool
trainpoints = dtrain[['Irrcal', 'uvIndex.1', 'cloudcover', 'humidity', 'tempC']]
testpoints= dtest[['Irrcal', 'uvIndex.1', 'cloudcover', 'humidity', 'tempC']]
trainpoints = trainpoints.values
testpoints = testpoints.values
results = []
minte = []
maxte = []
meante = []
#The below function parallelizes the computation of the distance between every test point and all the training points.
def parallel_dist_compute(i):
    temp_distances = []
    for j in range(0,trainpoints.shape[0]):
        temp_distances.append((euclidean(trainpoints[j],testpoints[i]), i, j))
    temp_distances.sort()
    corresponding_dc_power = []
    for k in range(0,50):
        corresponding_dc_power.append(dtrain.loc[temp_distances[k][2]]['dc_pow'])
    corresponding_dc_power.sort()
    corresponding_dc_power = corresponding_dc_power[10:]#This is done to remove the top and bottom 10 from the sorted distances list to deal with outliers
    corresponding_dc_power = corresponding_dc_power[:-10]
    print(i)
    return (i,corresponding_dc_power[0],corresponding_dc_power[-1],np.array(corresponding_dc_power).mean())


args = list( range(0,testpoints.shape[0],1))
p = Pool(8)
results = p.map(parallel_dist_compute, args)
results.sort()
for i in range(0,testpoints.shape[0],1):
    minte.append(results[i][1])
    maxte.append(results[i][2])
    meante.append(results[i][3])

#The minimum, maximum and mean of the nearest training points is added to the testing dataset
dtest['min'] = minte
dtest['max'] = maxte
dtest['mean'] = meante
del results
mintr = []
maxtr = []
meantr = []
#This function parallelizes the computation of the distance between training points and training points. 
def par_comp(i):
    temp_distances = []
    for j in range(0,trainpoints.shape[0]):
        temp_distances.append((euclidean(trainpoints[i],trainpoints[j]), i, j))
    temp_distances.sort()
    corresponding_dc_power = []
    for k in range(1,51):
        corresponding_dc_power.append(dtrain.loc[temp_distances[k][2]]['dc_pow'])
    corresponding_dc_power.sort()
    corresponding_dc_power = corresponding_dc_power[10:]
    corresponding_dc_power = corresponding_dc_power[:-10]
    mintr.append(corresponding_dc_power[0])
    maxtr.append(corresponding_dc_power[-1])
    meantr.append(np.array(corresponding_dc_power).mean())
    print(i)
    return (i,corresponding_dc_power[0],corresponding_dc_power[-1],np.array(corresponding_dc_power).mean())


args = list( range(0,trainpoints.shape[0],1))
p = Pool(11)
results = p.map(par_comp, args)
results.sort()
for i in range(0,trainpoints.shape[0],1):
    mintr.append(results[i][1])
    maxtr.append(results[i][2])
    meantr.append(results[i][3])

#The minimum, maximum and mean of the nearest training points is added to the training dataset
dtrain['min'] = mintr
dtrain['max'] = maxtr
dtrain['mean'] = meantr
scalers = []
#Output Scalers for every hour of the day
for i in range(0,24):
    scalers.append(MinMaxScaler(feature_range=(0, 1)))

#Input scalers for every hour of the day
inscalers = []
for i in range(0,24):
    inscalers.append(MinMaxScaler(feature_range=(0, 1)))

#With y standing for outputs and x standing for inputs, all the data is scaled again and the scalers are saved.
ytrainhourly = [[] for i in range(0,24)]
yact = [[] for i in range(0,24)]
for i in range(max((dtrain.hour_of_day.min()),(dtest.hour_of_day.min())),min((dtrain.hour_of_day.max() + 1),(dtest.hour_of_day.max() + 1))):
    print(i)
    te = dtrain[dtrain.hour_of_day == i].copy()
    te[['dc_pow']] = scalers[i].fit_transform(te[['dc_pow']])
    ytrainhourly[i] = te['dc_pow'].values
    del te


ytesthourly = [[] for i in range(0,24)]
for i in range(max((dtrain.hour_of_day.min()),(dtest.hour_of_day.min())),min((dtrain.hour_of_day.max() + 1),(dtest.hour_of_day.max() + 1))):
    te = dtest[dtest.hour_of_day == i].copy()
    yact[i] = dtest[['dc_pow']].values
    te[['dc_pow']] = scalers[i].transform(te[['dc_pow']])
    ytesthourly[i] = te['dc_pow'].values
    del te


xtrainhourly = [[] for i in range(0,24)]
for i in range(max((dtrain.hour_of_day.min()),(dtest.hour_of_day.min())),min((dtrain.hour_of_day.max() + 1),(dtest.hour_of_day.max() + 1))):
    te = dtrain[dtrain.hour_of_day == i].copy()
    te = te.iloc[1:]
    te['prev'] = ytrainhourly[i][:-1]
    ytrainhourly[i] = ytrainhourly[i][1:]
    te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year']] = inscalers[i].fit_transform(te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year']])
    xtrainhourly[i] = te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year','prev']].values
    del te


xtesthourly = [[] for i in range(0,24)]
for i in range(max((dtrain.hour_of_day.min()),(dtest.hour_of_day.min())),min((dtrain.hour_of_day.max() + 1),(dtest.hour_of_day.max() + 1))):
    te = dtest[dtest.hour_of_day == i].copy()
    te = te.iloc[1:]
    te['prev'] = ytesthourly[i][:-1]
    ytesthourly[i] = ytesthourly[i][1:]
    te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year']] = inscalers[i].transform(te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year']])
    xtesthourly[i] = te[['Irrcal','uvIndex.1','cloudcover', 'humidity','tempC','min', 'max', 'mean','hour_of_day', 'day_of_year','prev']].values
    del te


#This is a sequential neural network to predict the hourly production of the PV power plant. The same model architecture is used in 24 unconncected sections to predict every hour of the day.
def baseline_model1():
    model = Sequential()
    model.add(Dense(4, input_dim = 11, kernel_initializer='normal', activation = 'relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model



import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
#The model is trained using the hourly input and output data. Several iteration are considered to avoid local minimas.The best is saved.
ypred = [[] for i in range(0,24)]
models = [0] * 24
for i in range(max((dtrain.hour_of_day.min()),(dtest.hour_of_day.min())),min((dtrain.hour_of_day.max() + 1),(dtest.hour_of_day.max() + 1))):
    err = 10000000
    saved = 0
    for j in range(0,2):
        es = EarlyStopping(monitor='val_loss', mode='min', patience=60, min_delta=0.0001, verbose=1,restore_best_weights=True)
        model = KerasRegressor(build_fn=baseline_model1,epochs = 10000, batch_size=2, verbose=1, callbacks=[es])
        his = model.fit(xtrainhourly[i],ytrainhourly[i],validation_data = (xtesthourly[i],ytesthourly[i]))
        if min(his.history['val_loss']) < err:
            saved = model
            err = min(his.history['val_loss'])
    ypred[i] = scalers[i].inverse_transform(np.array(saved.predict(xtesthourly[i])).reshape(-1,1))
    yact[i] = scalers[i].inverse_transform(np.array(ytesthourly[i]).reshape(-1,1))
    models[i] = saved
    del model
    del saved
    del es


#The predicted output is added to the input data to compare the actual and predicted values of production.
dtest.reset_index()
counter = [-1]*24
precol = []
for i in range(0,dtest.values.shape[0]):
    print(i)
    if counter[dtest.loc[i]['hour_of_day']] < 0:
        precol.append(0)
        counter[dtest.loc[i]['hour_of_day']] = counter[dtest.loc[i]['hour_of_day']] + 1
    else:
        precol.append(ypred[dtest.loc[i]['hour_of_day']][counter[dtest.loc[i]['hour_of_day']]])
        counter[dtest.loc[i]['hour_of_day']] = counter[dtest.loc[i]['hour_of_day']] + 1


pred = []
for i in range(0,len(precol)):
    if precol[i] == 0:
        pred.append(precol[i])
    else:
        pred.append(precol[i][0])

