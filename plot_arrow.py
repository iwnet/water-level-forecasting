import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import process_csv_pairs
import pandas as pd
import numpy as np
import seaborn as sns
import os
from datetime import datetime, timedelta

sns.set()

def plot_time_series_with_arrows(real_data, predictions, path="arrow.png"):
    plt.figure(figsize=(15, 6), dpi=500)
    data_dates = real_data.index
    # Plot the black line for the time series values
    plt.plot(real_data, color='black', linewidth=1, label='Data', marker='o')
    # Add red arrows from values to predictions    
    for i, (idx, pred) in enumerate(predictions.items()):
        # skip first and 0s
        if i==0:
            continue
        day_before = idx + timedelta(days=-1)
        val = real_data[day_before]
        arrow_start = (day_before, val)
        arrow_end = (idx, pred)
        if pred==0 or val == 0:
            continue
        plt.annotate('', arrow_end, arrow_start, arrowprops=dict(arrowstyle='->', color='r', linewidth='2'))
    # label for arrows
    plt.plot([], color='r', label='Next day prediction - GRU', marker='>')
    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel('Water level (cm)')
    plt.title('Daily predictions')

    # Show legend
    plt.legend()
    plt.xticks(data_dates, rotation=30)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.ylim(np.min([real_data[real_data != 0].min(), predictions[predictions != 0].min()]) - 10, np.max([real_data, predictions]) + 10)
    plt.xlim(data_dates[0] - timedelta(days=+1), data_dates[-1] + timedelta(days=+1))

    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.cla()

if __name__ == "__main__":
    # Path to specific configuration
    path = "results/example/Wildungsmauer worst place -40/LSTM/hidden_32/layers_1/predLen_5/inputLen_15/trainPeriod_6-months"
        
    file_list = []
    for dir in os.listdir(path):
        file_list.append((
            "{}/{}/test.csv".format(path, dir),
            "{}/{}/step_forecast.csv".format(path, dir),
        ))
    file_list = sorted(file_list)[0:5]
    true_values, predicted_values = process_csv_pairs(file_list)
    plot_time_series_with_arrows(true_values.iloc[:,0], predicted_values.iloc[:,0])