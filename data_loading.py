
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scale_time_series(data, new_min, new_max):
    # Calculate the current minimum and maximum values in the data
    current_min = np.min(data)
    current_max = np.max(data)

    # Scale the data to the new range
    scaled_data = ((data - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min

    return scaled_data

def readDataset(filename="datasets/water-levels/waterlevels.xlsx"):
    df = pd.read_excel(filename, sheet_name='Waterconditions_daily pivot', skiprows=9, index_col="Date", parse_dates=True)   
    return df

def readWeather(sheetName, filename="datasets/weather/weather-data.xlsx"):
    df = pd.read_excel(filename, sheet_name=sheetName, index_col="datetime", parse_dates=True)   
    return df


def preProcess(dataSeries, plotPath=None, 
                train_start_date="2017-01-01", train_end_date="2017-05-31",
                test_start_date="2017-06-01", test_end_date="2017-06-30",
                saveTrain=False, saveTest=False, weatherDf=None):
    """
    Values are the time series for that specific station identifier
    Arguments:
        saveTrain: plot train set for each dataset
        saveTest: plot test set for each dataset
    """  
   
    # TODO experiment with different train/test
    train, test = dataSeries[train_start_date : train_end_date], dataSeries[test_start_date : test_end_date]    
    weatherDf_train = None
    weatherDf_test = None
    if weatherDf is not None:
        weatherDf_train, weatherDf_test = weatherDf[train_start_date : train_end_date], weatherDf[test_start_date : test_end_date]
    else:
        weatherDf_train, weatherDf_test = None, None

    # Plot each data column
    if ((saveTest or saveTrain) and plotPath != None):
        plt.cla()
        if saveTrain:
            plt.plot(train, label='Train')
        if saveTest:
            plt.plot(test, label='Test', color='green')        
        plt.legend()
        plt.xticks(rotation=90)
        Path("{}/".format(plotPath)).mkdir(parents=True, exist_ok=True)
        print("Saving.. {}/data.png".format(plotPath))
        plt.savefig("{}/data.png".format(plotPath), bbox_inches="tight")
        plt.cla()
        train.to_csv("{}/train.csv".format(plotPath))
        test.to_csv("{}/test.csv".format(plotPath))
            
    return {
        'train_dataset': train,
        'test_dataset': test,
        'train_weather_dataset': weatherDf_train,
        'test_weather_dataset': weatherDf_test
    }
