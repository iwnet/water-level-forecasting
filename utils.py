import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from os import cpu_count
from warnings import catch_warnings, filterwarnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

######################
# Utility
######################

stationNames = ["Kienstock worst place -40", "Pfelling -130", "Wildungsmauer worst place -40"]

def weighted_loss(y_true, y_pred):
    # Define a weighting scheme (e.g., exponential decay)
    # weights = np.exp(-np.arange(len(y_true)))
    # Linear decay
    weights = np.concatenate([1.0 - np.linspace(0, 0.3, int(np.floor(len(y_true)*0.3))), (1.0 - np.linspace(0.7, 1.0, int(np.ceil(len(y_true)*0.7))))])
    # Calculate residuals and take absolute values
    residuals = np.abs(y_true - y_pred)

    # Calculate weighted residuals
    weighted_residuals = residuals * weights

    # Calculate the loss as the sum of weighted residuals
    loss = np.sum(weighted_residuals)
    
    return loss



def mean_absolute_percentage_error(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    # Calculate the absolute percentage error for each pair of true and predicted values
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    # Calculate the mean of absolute percentage errors and convert it to a percentage
    mape = np.mean(absolute_percentage_error) * 100.0
    return mape

# Calculate rmse
def measure_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    return rmse


def test_stationarity(s):
    # Plot the time series
    s.plot()
    plt.title('Time Series')
    plt.savefig("data.png")
    plt.cla()
    # Calculate rolling mean and standard deviation
    rolmean = s.rolling(window=7).mean() # Change window size as needed
    rolstd = s.rolling(window=7).std()  # Change window size as needed

    # Plot the rolling mean and standard deviation
    plt.plot(s, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.savefig("meain_std.png")
    plt.cla()
    # Perform the ADF test
    print('Results of Augmented Dickey-Fuller Test:')
    adf_test = adfuller(s)
    print(f'ADF Statistic: {adf_test[0]}')
    print(f'p-value: {adf_test[1]}')
    print(f'Critical Values: {adf_test[4]}')

    # Plot the ACF and PACF
    plot_acf(s, lags=50)
    plt.savefig("acf.png")
    plt.cla()
    plot_pacf(s, lags=50)
    plt.savefig("pacf.png")
    plt.cla()
    
# score a model, return None on failure
def score_model(model, cfg, debug=False):
    rmse = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        predictions = model.forecast(cfg)
        rmse = measure_rmse(model.test_data, predictions)
    else:
    # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                predictions = model.forecast(cfg)
                rmse = measure_rmse(model.test_data, predictions)
        except:
            error = None
    # check for an interesting result
    # if rmse is not None:
    #     print(' > Model[%s] %.3f' % (key, rmse))
    return (key, rmse, cfg)


# grid search configs
def grid_search(model, cfg_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(model, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(model, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def process_csv_pairs(file_list):
    true_values = pd.DataFrame()
    predicted_values = pd.DataFrame()

    for file_pair in file_list:
        true_values = pd.concat([true_values, (pd.read_csv(file_pair[0], index_col=0, parse_dates=True))])
        true_values = true_values[~true_values.index.duplicated(keep='last')]
        predicted_values = pd.concat([predicted_values, pd.read_csv(file_pair[1], index_col=0, parse_dates=True, skiprows=1, header=None)])        
        predicted_values = predicted_values[~predicted_values.index.duplicated(keep='last')]

        
    true_values.sort_index(inplace=True)
    predicted_values.sort_index(inplace=True)
    # fill with missing dates
    idx = pd.date_range(true_values.index[0], true_values.index[-1], freq='D')
    true_values = true_values.reindex(idx, fill_value=0)
    predicted_values = predicted_values.reindex(idx, fill_value=0)
    return true_values, predicted_values


# Function to create a legend handler for arrows
class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Create a new arrow with the same properties as the original one
        p = FancyArrowPatch((0.5, 0.5), (1.0, 0.5), color=orig_handle.get_edgecolor(),
                            arrowstyle=orig_handle.get_arrowstyle(),
                            mutation_scale=orig_handle.get_mutation_scale(),
                            linestyle=orig_handle.get_linestyle(),
                            linewidth=orig_handle.get_linewidth(),
                            facecolor=orig_handle.get_facecolor(),
                            edgecolor=orig_handle.get_edgecolor(),
                            hatch=orig_handle.get_hatch(),
                            alpha=orig_handle.get_alpha())

        # Set the transform for the arrow
        p.set_transform(trans)

        return [p]