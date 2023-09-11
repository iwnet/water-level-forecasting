import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from utils import stationNames

sns.set()

parentPath = os.getcwd()
plot_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcb`d22',  # yellow-green
    '#17becf'   # cyan
]
plot_markers = [
    's',
    'd',
    'X',
]
def plot_against_real(real_data, predictions, path=""):

    plt.figure(figsize=(10, 6))    
    data_dates = pd.date_range(start=real_data.index[0], end=real_data.index[-1], freq='D')

    # Plot the black line for the last 21 days of training data
    plt.plot(real_data, color='black', linewidth=2, label='Data', marker='o')

    # Plot the red line for the test predictions
    plt.plot(predictions, color='red', linewidth=2, label='Predictions', marker='x')

    # Set labels and title
    plt.xlabel('Date')
    plt.xticks(data_dates, rotation=90)
    plt.ylabel('Value')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    if path == "":
        path = "data_and_predictions.png"
    plt.tight_layout()
    plt.savefig(path, format="png", bbox_inches="tight")
    plt.cla()

def main():

    kienstock = [
        "results/example/path/to/configurations",
        "results/example/path/to/configurations",
        "results/example/path/to/configurations",
    ]
    font_properties = {'family': 'sans-serif', 'size': 16, 'weight': 'normal'}

    paths = kienstock

    file_dict = {}
    for path in paths:
        dirList = path.split("/")
        stationName = [i for i in stationNames if i in dirList][0]
        modelName = dirList[dirList.index(stationName)+1] + "(" + dirList[dirList.index(stationName)+2].replace("hidden_","") + ")"
        if "continues-noweather" in dirList:
            modelName += " No weather data"
        for dir in os.listdir(path):
            pair = (
                "{}/{}/test.csv".format(path, dir),
                "{}/{}/one_forecast.csv".format(path, dir), 
            )
            file_dict.setdefault(modelName, []).append(pair)

    # keep only a few months, not the whole dataset
    for key in file_dict:
        idx = 0
        file_dict[key] = sorted(file_dict[key])[idx:idx+4]
    
    plt.figure(figsize=(15, 6), dpi=500)

    labels = []
    for i, (modelName, file_list) in enumerate(file_dict.items()):
        for file_pair in file_list:
            true_values = pd.read_csv(file_pair[0], index_col=0, parse_dates=True)
            predicted_values = pd.read_csv(file_pair[1], index_col=0, skiprows=1, header=None, parse_dates=True)
            plt.plot(true_values, color='black', marker='o', linewidth=2, label='Water level measurements')
            plt.axvline(x=predicted_values.index[0], color='r', linestyle='--', label='Re-train and forecast')
            if (predicted_values.index[0] not in labels):
                labels.append(predicted_values.index[0])
            plt.plot(predicted_values, color=plot_colors[i], marker=plot_markers[i], linewidth=2, label=modelName)
    plt.xlabel('Date', font_properties=font_properties)
    
    plt.xticks(labels, font_properties=font_properties)    
    plt.yticks(font_properties=font_properties)
    plt.ylabel('Water level (cm)', font_properties=font_properties)
    
    plt.title(stationName)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=font_properties['size'])

    # Show the plot
        
    plt.tight_layout()
    plt.savefig("long_forecasts.png", format="png", bbox_inches="tight")
    plt.cla()

if __name__ == "__main__":
    main()

