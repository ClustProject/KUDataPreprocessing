
"""Seasonal Trend Decomposition and Prediction module for Numeric dataset. 

"""

import statsmodels.api as sm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse
from pmdarima.arima import auto_arima

parser = argparse.ArgumentParser(description='Temporal One-class Anomaly Detection')
parser.add_argument('--data_path', type=str, default='./dataset/machine_temperature_system_failure.csv')
parser.add_argument('--seasonal_output_path', type=str, default='./dataset/seasonal_decomposed.csv')
parser.add_argument('--trend_output_path', type=str, default='./dataset/trend_decomposed.csv')

class DataDecomposer():
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.dataset['timestamp'] = pd.to_datetime(self.dataset['timestamp'])
        self.dataset = self.dataset.set_index('timestamp')
        
    def STL_decomposition(self, model='additive', period = 288):  
        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.DecomposeResult.html#statsmodels.tsa.seasonal.DecomposeResult
        decompostion = sm.tsa.seasonal_decompose(self.dataset['value'],  model=model, period=period)
        fig = decompostion.plot()
        fig.set_size_inches(12,10)
        plt.show()
        
        seasonal = decompostion.seasonal
        trend = decompostion.trend
        trend = trend.dropna()
    
        return seasonal, trend    
    
    def AutoArima(self, time_type):
        if time_type == 'seasonal':
            train_data = seasonal.copy().iloc[:-288]
            test_data =  seasonal.copy().iloc[-288:]
        elif time_type == 'trend':
            train_data = trend.copy().iloc[:-288]
            test_data = trend.copy().iloc[-288:]
            
        auto_arima_model = auto_arima(train_data, seasonal=True, 
                                      trace=True, d=1, D=1, 
                                      error_action='ignore',  
                                      suppress_warnings=True, 
                                      stepwise=False,
                                      n_jobs=8)

    
        auto_arima_model.summary()
        prediction = auto_arima_model.predict(288, return_conf_int=True)

        predicted_value = prediction[0]
        predicted_ub = prediction[1][:,0]
        predicted_lb = prediction[1][:,1]
        predict_index = list(test_data.index)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        train_data.plot(ax=ax);
        ymin, ymax = ax.get_ylim()
        ax.vlines(predict_index[0], ymin, ymax, linestyle='--', color='r', label='Start of Forecast');
        ax.plot(predict_index, predicted_value, label = 'Prediction')
        ax.fill_between(predict_index, predicted_lb, predicted_ub, color = 'k', alpha = 0.1, label='0.95 Prediction Interval')
        ax.legend(loc='upper left')
        plt.suptitle(f'{time_type} SARIMA {auto_arima_model.order},{auto_arima_model.seasonal_order}')
        plt.show()
            
        return auto_arima_model, prediction
    
def main(args):
    dataDecomposer = DataDecomposer(args.data_path)
    seasonal, trend = dataDecomposer.STL_decomposition()
    
    model_seasonal, pred_seasonal = dataDecomposer.AutoArima(time_type = 'seasonal')
    model_trend, pred_trend = dataDecomposer.AutoArima(time_type = 'trend')
    
    seasonal.to_csv(args.seasonal_output_path, index=False, header=False)
    trend.to_csv(args.trend_output_path, index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
