#!/usr/bin/python3

######################
# Comment section
######################


import os
import pandas as pd
import numpy as np
from pathlib import Path
import data_loading
import models
from timeframe_gen import generate_date_windows 

######################
# Main
######################
def main():
    stations = [
        'Wildungsmauer worst place -40',
        'Kienstock worst place -40',
        'Pfelling -130',
    ]

    df = data_loading.readDataset()

    # Create list of model configurations
    modelList = []
    rnnModels = ["LSTM", "GRU", "RNN"]
    hidden_dims = [32, 64, 128]
    n_rnn_layers = [1, 2]
    prediction_lengths = [5, 10]
    training_lengths = [15, 21, 32] 
    for model in rnnModels:
        for n_rnn_layer in n_rnn_layers:
            for hidden_dim in hidden_dims:
                for prediction_length in prediction_lengths:                        
                    for training_length in training_lengths:
                        config = {
                            "model": model,
                            "hidden_dim": hidden_dim,
                            "n_rnn_layers": n_rnn_layer,
                            "training_length": training_length + prediction_length,
                            "input_chunk_length": training_length,
                            "epochs": 50
                        }
                        modelList.append(config)    
    for station in stations:
        weatherDf = data_loading.readWeather(sheetName=station)
        series = df[station]

        for conf in modelList:
            train_window = 6
            timeframes = generate_date_windows("2017-01-01", "2020-12-31", month_training_window=train_window)
            for timeframe in timeframes:
                # Output path
                outputPath = "results/example/{}/{}/hidden_{}/layers_{}/predLen_{}/inputLen_{}/trainPeriod_{}-months/train-from-{}_to-{}".format(
                    series.name, conf["model"], 
                    conf["hidden_dim"], 
                    conf["n_rnn_layers"], 
                    conf["training_length"] - conf["input_chunk_length"],
                    conf["input_chunk_length"], 
                    train_window,
                    timeframe["train_start_date"], 
                    timeframe["train_end_date"])
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath)

                train_start_date, train_end_date = timeframe["train_start_date"], timeframe["train_end_date"]
                test_start_date, test_end_date = timeframe["test_start_date"], timeframe["test_end_date"]

                print("Pre processing...")
                datasets = data_loading.preProcess(series, outputPath,
                    train_start_date=train_start_date, train_end_date=train_end_date, 
                    test_start_date=test_start_date, test_end_date=test_end_date, 
                    saveTest=True, saveTrain=True, weatherDf=weatherDf)
                
                train, test = datasets['train_dataset'], datasets['test_dataset']
                weatherTrain = datasets['train_weather_dataset']
                weatherTest = datasets['test_weather_dataset']
                
                print("Creating Models...")
                models.RNNModelTrainer(train, test, outputPath,
                        pd.concat([
                            weatherTrain['precipcover'],
                            weatherTest['precipcover']],
                        ), config=conf,
                        earlyStop=False
                ).Fit()
    return



if __name__ == "__main__":
    main()


