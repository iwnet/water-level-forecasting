import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import process_csv_pairs


def create_scatter_plot(true_values, predicted_values):
    """
    Create a scatter plot for true values vs predicted values.

    Parameters:
    - true_values: List or array of true values.
    - predicted_values: List or array of predicted values.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot([100, max(true_values.values)], [100, max(predicted_values.values)], color='red', linestyle='--', label='Diagonal Line')

    plt.title('Scatter Plot of True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.grid(True)
    plt.savefig("scatter.png")

if __name__ == "__main__":
    # Path to specific configuration
    path = "results/example/Pfelling -130/GRU/hidden_2048/layers_1/predLen_1/inputLen_10/trainPeriod_6-months"
    file_list = []
    for dir in os.listdir(path):
        file_list.append((
            "{}/{}/test.csv".format(path, dir),
            "{}/{}/step_forecast.csv".format(path, dir), 
        ))


    test_values, predicted_values = process_csv_pairs(file_list)

    create_scatter_plot(test_values, predicted_values)


