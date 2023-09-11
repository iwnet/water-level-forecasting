import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import numpy as np
from plot_long_forecast import plot_against_real
from plot_arrow import plot_time_series_with_arrows


from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from abc import ABC, abstractmethod
from pytorch_lightning.callbacks import Callback
from torch.nn import MSELoss

from utils import grid_search, measure_rmse

from darts.models import RNNModel
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping


sns.set()

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:        
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))


######################
# Generic Trainer
######################
class ITrainer(ABC):

    @abstractmethod
    def forecast(self, cfg):
        pass
    @abstractmethod
    def Fit(self):
        pass


######################
# Auto regression
######################

class AutoRegTrainer(ITrainer):
    def __init__(self, train, test, plotPath):
        self.train_data = train
        self.test_data = test
        self.plotPath = plotPath
        return
    
    def forecast(self, cfg):
        totalLen = len(self.train_data) + len(self.test_data)

        AutoRegModel = AutoReg(self.train_data.values.astype(float), lags=cfg)
        AutoRegModel = AutoRegModel.fit()

        predictions_df = pd.Series(AutoRegModel.predict(start = len(self.train_data), end = totalLen - 1))
        predictions_df.index = self.test_data.index
        
        return predictions_df

    def Fit(self, lags=[], parallel=True):    
        totalLen = len(self.train_data) + len(self.test_data)

        # Data is daily. Use backestep daily, weekly, monthly, yearly
        if lags == []:
            configuration_lags_list = [i for i in range(365)]
        else:
            configuration_lags_list = lags
        
        print("Searching best Autoregression model...")
        scores = grid_search(self, configuration_lags_list)
        best_AutoReg = scores[0]
        best_lag = best_AutoReg[2]
        print("Best AutoRegression model with lag: ", best_AutoReg)
        
        AutoRegModel = AutoReg(self.train_data.values.astype(float), lags=best_lag)
        AutoRegModel = AutoRegModel.fit()

        predictions_df = pd.Series(AutoRegModel.predict(start = len(self.train_data), end = totalLen - 1))
        predictions_df.index = self.test_data.index

        rmse = measure_rmse(self.test_data, predictions_df)
        print("AutoReg RMSE: ", rmse)

        plt.cla()
        plt.plot(self.test_data, color='green', label = 'Test')
        plt.plot(predictions_df, color='red', label = 'AutoReg')
        
        plt.legend(title="RMSE: {}".format(round(rmse, 3)))
        Path(self.plotPath).mkdir(parents=True, exist_ok=True)
        plt.title("AutoRegression model (lags={})".format(best_lag))
        print("Saving {}/AutoReg.png".format(self.plotPath))
        plt.xticks(rotation=90)
        plt.savefig("{}/AutoReg.png".format(self.plotPath), bbox_inches="tight")

        fp = open("{}/AutoRegPredictions.csv".format(self.plotPath), 'w')
        fp.write(predictions_df.to_string() + "\n")
        fp.close()
        

######################
# ARIMA
######################
class ARIMATrainer(ITrainer):
    def __init__(self, train, test, plotPath):
        self.train_data = train
        self.test_data = test
        self.plotPath = plotPath
        return

    # create a set of sarima configs to try
    def arima_configs(self):
        models = list()
        # define config lists
        p_params = [1, 2, 3]
        d_params = [1, 2]
        q_params = [1, 2, 3, 4]
        P_params = [0, 1]
        D_params = [1]
        Q_params = [0, 1]
        m_params = [7, 30]
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p,d,q), (P,D,Q,m)]
                                        models.append(cfg)
        return models

    def forecast(self, cfg):
        order, sorder = cfg
        ARIMAmodel = ARIMA(self.train_data, order = order, seasonal_order=sorder)
        ARIMAmodel = ARIMAmodel.fit()
        
        y_pred = ARIMAmodel.get_forecast(steps=len(self.test_data.index))
        return y_pred.predicted_mean

    def Fit(self):
        conf_list = self.arima_configs()
        
        scores = grid_search(self, conf_list)
        best_arima = scores[0]
        print("Best ARIMA: ", best_arima)
        predictions = self.forecast(best_arima[2])

        rmse = measure_rmse(self.test_data.values, predictions)
        print("ARIMA RMSE: ", rmse)
                
        plt.cla()
        plt.plot(self.test_data, color='green', label='Test')
        plt.plot(predictions, label='ARIMA', color='r')
        
        plt.legend(title="RMSE: {}".format(round(rmse, 3)))
        plt.title("ARIMAX {}".format(str(best_arima[0]).replace(' ','')))
        print("Saving {}/ARIMA.png".format(self.plotPath))
        plt.xticks(rotation=90)
        plt.savefig("{}/ARIMA.png".format(self.plotPath), bbox_inches="tight")
        
        fp = open("{}/ARIMAPredictions.csv".format(self.plotPath), 'w')
        fp.write(predictions.to_string() + "\n")
        fp.close()
        return
    


######################
# Auto ARIMA
######################
class AutoARIMATrainer(ITrainer):
    def __init__(self, train, test, plotPath):
        self.train_data = train
        self.test_data = test
        self.plotPath = plotPath
        self.model = None
        return
    
    def forecast(self, cfg):
        if self.model is not None:
            return pd.Series(self.model.predict(n_periods=len(self.test_data)), index=self.test_data.index, name="Predictions")
        else:
            raise Exception("model not initialized")
        
    def Fit(self):    
        self.model = auto_arima(self.train_data.values, 
                            start_p=2, start_q=0,
                            max_p=4, max_q=30,
                            d=1, max_d=1,
                            start_P=0, max_P=4,
                            D=1, max_D=1,
                            start_Q=0, max_Q=30,
                            m=30, seasonal=True, error_action='warn', trace=True,
                            suppress_warnings=True, stepwise=True, n_fits=200,
                            n_jobs=-1)

        predictions = pd.Series(self.model.predict(n_periods=len(self.test_data)), index=self.test_data.index, name="Predictions")

        rmse = measure_rmse(self.test_data, predictions)
        print("Auto ARIMA RMSE: ", rmse)

        plt.cla()
        plt.plot(self.test_data, color='green', label='Test')
        plt.plot(predictions, color='red', label='Auto ARIMA')
        plt.legend(title="RMSE: {}".format(round(rmse, 3)))
        plt.title("AutoArima: {}".format(str(self.model)))
        print("Saving {}/AutoARIMA.png".format(self.plotPath))
        plt.xticks(rotation=90)
        plt.savefig("{}/AutoARIMA.png".format(self.plotPath), bbox_inches="tight")

        
        fp = open("{}/AutoARIMAPredictions.csv".format(self.plotPath), 'w')
        fp.write(predictions.to_string() + "\n")
        fp.close()
        return


######################
# Seasonal ARIMA (SARIMA)
######################
class SARIMATrainer(ITrainer):
    def __init__(self, train, test, plotPath, exogenousTrain=None, exogenousTest=None):
        self.train_data = train
        self.test_data = test
        self.plotPath = plotPath
        self.exog_train=exogenousTrain
        self.exog_test=exogenousTest
        return
    
    def forecast(self, cfg):
        order, sorder, trend = cfg
        SARIMAXmodel = SARIMAX(self.train_data, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertability=False, exog=self.exog_train)
        # startParams = [0, 0, 0, 0, 1]
        # if self.exog_train is not None:
        #     for i in self.exog_train: startParams = [0] + startParams
        SARIMAXmodel = SARIMAXmodel.fit( disp=False)

        
        y_pred = SARIMAXmodel.get_forecast(steps=len(self.test_data.index), exog=self.exog_test)
        
        return y_pred.predicted_mean
  
    # create a set of sarima configs to try
    def sarima_configs(self):
        models = list()
        # define config lists
        p_params = [3, 4, 20]
        d_params = [1]
        q_params = [0, 1, 2, 7]
        t_params = ['n', 'c']
        P_params = [0, 1, 2, 3, 4]
        D_params = [1]
        Q_params = [0, 1, 2, 3]
        m_params = [7, 30]
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for t in t_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p,d,q), (P,D,Q,m), t]
                                        models.append(cfg)
        return models

    def Fit(self):

        print("Season arima search....")
        conf_list = self.sarima_configs()
            
        scores = grid_search(self, conf_list)
        best_sarima = scores[0]
        print("Best SARIMAX: ", best_sarima)
        predictions = self.forecast(best_sarima[2])
        
        rmse = measure_rmse(self.test_data.values, predictions)
        print("SARIMA RMSE: ", rmse)

        plt.cla()
        plt.plot(self.test_data, color='green', label='Test')
        plt.plot(predictions, label='SARIMA', color='r')
        
        plt.legend(title="RMSE: {}".format(round(rmse, 3)))
        plt.title("SARIMAX {}".format(str(best_sarima[0]).replace(' ','')))
        print("Saving {}/SARIMAX.png".format(self.plotPath))
        plt.xticks(rotation=90)
        plt.savefig("{}/SARIMAX.png".format(self.plotPath), bbox_inches="tight")
        
        fp = open("{}/SARIMAXPredictions.csv".format(self.plotPath), 'w')
        fp.write(predictions.to_string() + "\n")
        fp.close()
        return


######################
# Dart RNN
######################


class RNNModelTrainer(ITrainer):
    def __init__(self, train: TimeSeries, test: TimeSeries, plotPath, future_covariates=None, earlyStop=False, config=None):
        self.model_config = config
        self.stepForecast = True
        self.earlyStop = earlyStop
        # Normalize the time series (note: we avoid fitting the transformer on the validation set)
        self.transformer = Scaler()
        self.train_data = train
        self.test_data = test
        self.scaled_train_data, self.scaled_val_data = self.transformer.fit_transform(TimeSeries.from_series(self.train_data)).split_after(0.75)
        self.scaled_test_data = self.transformer.transform(TimeSeries.from_series(test))
        
        # self.train_data = TimeSeries.from_series(train)
        # self.test_data = TimeSeries.from_series(test)
        self.plotPath = plotPath
        if future_covariates is not None:
            self.covariates = self.transformer.transform(TimeSeries.from_series(future_covariates))        
        else:
            self.covariates = None
        
        self.init_model()
        return
    
    def init_model(self):
        config = self.model_config
        pl_trainer_kwargs = {}
        self.loss_logger = LossLogger()

        pl_trainer_kwargs["callbacks"] = [self.loss_logger]
        if self.earlyStop:
            my_stopper = EarlyStopping(
                monitor="val_loss",  # "val_loss",
                patience=20,
                min_delta=0.0000001,
                mode='min',
            )
            pl_trainer_kwargs["callbacks"].append(my_stopper)

        if config is not None:
            self.epochs = config["epochs"]
            self.model = RNNModel(
                        model=config["model"],
                        hidden_dim=config["hidden_dim"],
                        n_rnn_layers=config["n_rnn_layers"],
                        input_chunk_length=config["input_chunk_length"],
                        training_length=config["training_length"],
                        n_epochs=config["epochs"],
                        nr_epochs_val_period=1,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                    )
            results = self.model.lr_find(series=self.scaled_train_data, val_series=self.scaled_val_data, future_covariates=self.covariates, val_future_covariates=self.covariates)            
            results.plot(suggest=True)
            plt.savefig("{}/lr_find.png".format(self.plotPath))
            plt.cla()
            self.model = RNNModel(
                        model=config["model"],
                        hidden_dim=config["hidden_dim"],
                        n_rnn_layers=config["n_rnn_layers"],
                        input_chunk_length=config["input_chunk_length"],
                        training_length=config["training_length"],
                        n_epochs=config["epochs"],
                        nr_epochs_val_period=1,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                        loss_fn=MSELoss(),
                        optimizer_kwargs={"lr": results.suggestion()}
                    )
        else:
            self.model = RNNModel(
                            model="LSTM",
                            hidden_dim=64,
                            # dropout=0.3,
                            n_rnn_layers=2,
                            input_chunk_length=20,
                            training_length=50,
                            n_epochs=200,
                            pl_trainer_kwargs=pl_trainer_kwargs
                        )
            self.epochs = 200
        
    def forecast(self, cfg):
        return self.model.predict(len(self.scaled_test_data), series=self.scaled_train_data, future_covariates=self.covariates)        
        
    
    def Fit(self):
        # Make sure the model is actually trained - otherwise re train (probably lr_find failed to find a good learning rate)
        while True:
            self.model.fit(series=self.scaled_train_data, val_series=self.scaled_val_data, val_future_covariates=self.covariates, future_covariates=self.covariates, epochs=self.epochs)
            if (np.mean(self.loss_logger.val_loss[:5]) < np.mean(self.loss_logger.val_loss[-5:])*2.0):
                self.init_model()
            else:
                break

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_logger.val_loss, label="val loss")
        plt.plot(self.loss_logger.train_loss, label="train loss")
        plt.legend()
        plt.savefig("{}/loss.png".format(self.plotPath))
        plt.cla()
        
        # Make daily predictions while updating the input.
        predictions = []
        input = self.scaled_train_data.concatenate(self.scaled_val_data)
        for i in range(len(self.scaled_test_data)):
            predictions.append(self.model.predict(1, series=input, future_covariates=self.covariates).values().flatten()[0])
            input = input.append_values(self.scaled_test_data[i].values().flatten())
            
        # Create series
        predictions = TimeSeries.from_values(np.array(predictions))
        
        reversed_scaled_predictions = self.transformer.inverse_transform(predictions).pd_series()
        reversed_scaled_predictions.index = self.test_data.index
        
        # Plot daily predictions
        plot_time_series_with_arrows(self.test_data, reversed_scaled_predictions, path="{}/step_forecast.png".format(self.plotPath))
        reversed_scaled_predictions.to_csv("{}/step_forecast.csv".format(self.plotPath))

        # Make long forecast and plot result.
        predictions = self.model.predict(len(self.scaled_test_data), series=self.scaled_train_data.concatenate(self.scaled_val_data), future_covariates=self.covariates)
        reversed_scaled_predictions = self.transformer.inverse_transform(predictions).pd_series()
        plot_against_real(self.test_data, reversed_scaled_predictions, "{}/one_forecast.png".format(self.plotPath))
        reversed_scaled_predictions.to_csv("{}/one_forecast.csv".format(self.plotPath))
