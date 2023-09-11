import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import measure_rmse
# Function to calculate RMSE

# Function to read CSV pairs and calculate RMSE for each day
def process_csv_pairs(file_list):
    results = []
    for file_pair in file_list:
        # Assuming file_pair is a tuple with (true_values_file, predicted_values_file)
        true_values = pd.read_csv(file_pair[0], index_col=0)
        predicted_values = pd.read_csv(file_pair[1], index_col=0, skiprows=1, header=None)

        rmse_values = [measure_rmse(true_values.values[i], predicted_values.values[i]) for i in range(len(true_values))]       
        results.append(rmse_values)

    return results

def main():
    # Example file list
    path = "results/example/path/to/configuration"
    file_list = []
    for dir in os.listdir(path):
        file_list.append((
            "{}/{}/test.csv".format(path, dir),
            "{}/{}/one_forecast.csv".format(path, dir), 
            "{}/{}/train.csv".format(path, dir), 
        ))
        
    # Process CSV pairs
    rmse_results = process_csv_pairs(file_list)
    font_properties = {'family': 'sans-serif', 'size': 20, 'weight': 'normal'}
    # Convert results to a DataFrame for easier plotting
    df_rmse = pd.DataFrame(rmse_results, columns=[i+1 for i in range(len(rmse_results[0]))])
    # Plotting
    plt.figure(figsize=(10, 7), dpi=500)
    sns.violinplot(data=df_rmse, inner='box', palette='husl', cut=0)
    # plt.title('Distribution of RMSE for Each Day - Kienstock GRU(32) no weather data')
    plt.xlabel('Day', font_properties=font_properties)
    plt.xticks(rotation=0, font_properties=font_properties)
    plt.ylabel('Prediction difference (in cm)', font_properties=font_properties)
    plt.ylim(0, 300)
    plt.yticks(np.arange(0, 300, 25)[1:], font_properties=font_properties)
    plt.tight_layout()
    plt.savefig("rmse_violin.png", format="png", bbox_inches="tight")
    plt.cla()


if __name__ == "__main__":
    main()