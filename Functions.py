import pandas as pd
import numpy as np

# import data
# create training sets and test sets
def import_data():
    file_names = [
        'Commercial1.csv', 'Commercial2.csv', 'Commercial3.csv', 'Commercial4.csv', 'Commercial5.csv',
        'Commercial6.csv', 'Commercial7.csv', 'Commercial8.csv', 'Commercial9.csv', 'Commercial10.csv',
        'Hospital1.csv', 'Hospital2.csv', 'Hospital3.csv', 
        'Office1.csv', 'Office2.csv', 'Office3.csv', 'Office4.csv', 'Office5.csv',
        'Sport1.csv', 'Sport2.csv', 'Sport3.csv', 
        'Tourist1.csv', 'Tourist2.csv', 'Tourist3.csv', 'Tourist4.csv',
        'Residential1.csv', 'Residential2.csv', 'Residential3.csv', 'Residential4.csv', 'Residential5.csv'
    ]

    X = []
    for file in file_names:
        dataset = pd.read_csv(f'D:/Edge/ECE1724F/Project/Reference/nh2358f5mf-4/nh2358f5mf-4/dataset/{file}')
        rate_values = dataset['RATE'].values.astype('float32')
        X.append(rate_values)

    train_size = int(0.8 * len(X[0]))
    test_size = len(X[0]) - train_size

    return X, train_size, test_size


# trend feature
def create_interval_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)



# cycle feature
def create_cycle_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back * 2):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back:i + look_back + look_back])
    return np.asarray(dataX), np.asarray(dataY)



# RNN parameters
def create_RNNs_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back * 2):
        dataX.append(dataset[i:i + look_back])
        dataY.append(dataset[i + look_back + look_back])
    return np.asarray(dataX), np.asarray(dataY)


# import time-series-decomposition dataset
def import_TSD_data():
    file_names = [
        'Commercial1.csv', 'Commercial2.csv', 'Commercial3.csv', 'Commercial4.csv', 'Commercial5.csv',
        'Commercial6.csv', 'Commercial7.csv', 'Commercial8.csv', 'Commercial9.csv', 'Commercial10.csv',
        'Hospital1.csv', 'Hospital2.csv', 'Hospital3.csv', 
        'Office1.csv', 'Office2.csv', 'Office3.csv', 'Office4.csv', 'Office5.csv',
        'Sport1.csv', 'Sport2.csv', 'Sport3.csv', 
        'Tourist1.csv', 'Tourist2.csv', 'Tourist3.csv', 'Tourist4.csv',
        'Residential1.csv', 'Residential2.csv', 'Residential3.csv', 'Residential4.csv', 'Residential5.csv'
    ]

    RATE, FS, INDICATOR = [], [], []
    for file in file_names:
        dataset = pd.read_csv(f'D:/Edge/ECE1724F/Final/Project_test/dataset/{file}')
        RATE.append(dataset['RATE'].values.astype('float32'))
        FS.append(dataset['FS'].values.astype('float32')) # Cycle features after FT
        INDICATOR.append(dataset['INDICATOR'].values.astype('float32'))

    train_size = int(0.8 * len(RATE[0]))
    test_size = len(RATE[0]) - train_size

    return RATE, FS, INDICATOR, train_size, test_size
