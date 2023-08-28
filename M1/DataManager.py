import os
import numpy as np
import pandas as pd
import pendulum as time
import urllib3 as request
import urllib3.exceptions
import yfinance as finance
import matplotlib.pyplot as plt

from tensorflow import keras
from joblib import dump, load
from colorama import Fore, Style
from Wavelet_denoising import tsd
from sklearn.preprocessing import MinMaxScaler


class DataManager:

    def __init__(self, model_config, tickers) -> None:
        self.__tickers = tickers
        self.__config = model_config
        self.__cached_tickers = {}
        self.__test_runs_periods = {}
        self.__test_runs_dataframes = {}
        self.__scalers = {}
        self.__datasets = {}

        if tickers is None or len(tickers) == 0:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] Cannot initialize DataManager without tickers. '
                  f'{Style.BRIGHT}')
            quit(1)

        for ticker in tickers:
            self.__cached_tickers.update({ticker: False})

    @property
    def tickers(self) -> list:
        return self.__tickers

    @property
    def dataframes(self) -> dict:
        return self.__test_runs_dataframes

    @property
    def datasets(self) -> dict:
        return self.__datasets

    @staticmethod
    def check_internet_connection() -> bool:
        try:
            response = (request.connection_from_url('https://www.google.com')
                        .request('GET', '/', timeout=1.0))
            if response.status == 200:
                return True
            else:
                return False
        except urllib3.exceptions.NewConnectionError:
            return False
        except urllib3.exceptions.MaxRetryError:
            return False

    @staticmethod
    def df_to_timeseries(dataframe) -> tuple:
        timeseries = keras.utils.timeseries_dataset_from_array(
            dataframe.values.ravel(),
            None,
            sequence_length=20,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False
        )
        dataset = list(timeseries.as_numpy_iterator())
        dataset = np.concatenate(dataset, axis=0)
        return dataset[:, :-1], dataset[:, -1].reshape(-1, 1)

    @staticmethod
    def plot_data(title, training, validation, test, path):
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.plot(training, label='training_set')
        plt.plot(validation, label='validation_set')
        plt.plot(test, label='test_set')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{path}')
        plt.close()

    def init_periods(self) -> None:
        period_begin = time.from_format(self.__config.period_begin, 'YYYY-MM-DD', tz='Europe/Rome')
        for test_run in range(0, self.__config.tr_amt):
            if test_run != 0:
                period_begin = period_begin.add(months=self.__config.tr_step_size)
            training_end = period_begin.add(years=self.__config.tr_delay_y)
            validation_end = training_end.add(months=self.__config.tr_delay_m)
            period_end = validation_end.add(months=self.__config.tr_delay_m)
            self.__test_runs_periods.update({test_run: [period_begin, training_end, validation_end, period_end]})

    def import_dataframes(self) -> None:
        model_data_path = f'./data'
        if os.path.exists(model_data_path):
            imported_dataframes = 0
            for ticker in self.__tickers:
                ticker_data_path = f'{model_data_path}/{ticker}'
                if os.path.exists(ticker_data_path):
                    missing_test_runs = False
                    ticker_test_runs_dfs = {}
                    ticker_test_runs_scalers = {}
                    for test_run in self.__test_runs_periods.keys():
                        if not missing_test_runs:
                            training_csv_exists = os.path.exists(f'{ticker_data_path}/test_run_{test_run}_training.csv')
                            validation_csv_exists = os.path.exists(f'{ticker_data_path}/test_run_{test_run}_validation'
                                                                   f'.csv')
                            test_csv_exists = os.path.exists(f'{ticker_data_path}/test_run_{test_run}_test.csv')
                            scaler_exists = os.path.exists(f'{ticker_data_path}/test_run_{test_run}_scaler.joblib')
                            if training_csv_exists and validation_csv_exists and test_csv_exists and scaler_exists:
                                ticker_test_runs_dfs.update({
                                    test_run: {
                                        'training': pd.read_csv(f'{ticker_data_path}/test_run_{test_run}_training.csv',
                                                                index_col='Date', parse_dates=True),
                                        'validation': pd.read_csv(f'{ticker_data_path}/test_run_{test_run}'
                                                                  f'_validation.csv', index_col='Date',
                                                                  parse_dates=True),
                                        'test': pd.read_csv(f'{ticker_data_path}/test_run_{test_run}_test.csv',
                                                            index_col='Date', parse_dates=True)
                                    }
                                })
                                ticker_test_runs_scalers.update({test_run: load(f'{ticker_data_path}/'
                                                                                f'test_run_{test_run}_scaler.joblib')})
                            else:
                                missing_test_runs = True
                    if not missing_test_runs:
                        self.__test_runs_dataframes.update({ticker: ticker_test_runs_dfs})
                        self.__scalers.update({ticker: ticker_test_runs_scalers})
                        self.__cached_tickers.update({ticker: True})
                        imported_dataframes += 1
                        print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] Ticker {ticker} imported. '
                              f'{Style.RESET_ALL}')
                    else:
                        print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] Ticker {ticker} not imported due to '
                              f'missing data. {Style.RESET_ALL}')
                else:
                    print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] No local data found for ticker {ticker}. '
                          f'{Style.RESET_ALL}')
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] {imported_dataframes} out of '
                  f'{len(self.__tickers)} ticker(s) imported from local data. {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] No local data found for the model. '
                  f'{Style.RESET_ALL}')

    def check_dfs_availability(self) -> bool:
        if len(self.__test_runs_dataframes.keys()) == len(self.__tickers):
            return True
        else:
            return False

    def download_dataframes(self) -> None:
        if self.check_internet_connection():
            for ticker in self.__tickers:
                if (ticker not in self.__test_runs_dataframes.keys() and
                        not self.__cached_tickers.get(ticker)):
                    ticker_dataframe = finance.download(ticker, self.__test_runs_periods.get(0)[0],
                                                        self.__test_runs_periods.get(self.__config.tr_amt - 1)[3]
                                                        .add(days=1), progress=False)
                    ticker_dataframe = ticker_dataframe[['Adj Close']]
                    ticker_dataframe = ticker_dataframe.rename(columns={'Adj Close': 'adj_close'}, inplace=False)
                    png_export_path = f'./images/data-preprocessing/{ticker}'
                    if not os.path.exists(png_export_path):
                        os.makedirs(png_export_path)
                    ticker_test_runs_dfs = {}
                    for test_run in self.__test_runs_periods.keys():
                        ticker_dataframe_tmp = ticker_dataframe.copy(deep=True)
                        test_run_periods = self.__test_runs_periods.get(test_run)
                        ticker_test_runs_dfs.update({
                            test_run: {
                                'training': ticker_dataframe_tmp.loc[f'{test_run_periods[0].to_date_string()}':
                                                                     f'{test_run_periods[1].to_date_string()}'],
                                'validation': ticker_dataframe_tmp.loc[
                                              f'{test_run_periods[1].add(days=1).to_date_string()}':
                                              f'{test_run_periods[2].to_date_string()}'],
                                'test': ticker_dataframe_tmp.loc[f'{test_run_periods[2].add(days=1).to_date_string()}':
                                                                 f'{test_run_periods[3].to_date_string()}']
                            }
                        })
                        self.plot_data(f'{ticker} original data subset (test run {test_run})',
                                       ticker_test_runs_dfs.get(test_run).get('training'),
                                       ticker_test_runs_dfs.get(test_run).get('validation'),
                                       ticker_test_runs_dfs.get(test_run).get('test'),
                                       f'{png_export_path}/test_run_{test_run}_phase_1.png')
                    self.__test_runs_dataframes.update({ticker: ticker_test_runs_dfs})
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] Cannot download dataframes without internet '
                  f'connection. {Style.RESET_ALL}')

    def normalize_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                png_export_path = f'./images/data-preprocessing/{ticker}'
                if not os.path.exists(png_export_path):
                    os.makedirs(png_export_path)
                ticker_test_runs_scalers = {}
                for test_run in self.__test_runs_periods.keys():
                    scaler = MinMaxScaler(copy=False, clip=False)
                    scaler.fit(self.__test_runs_dataframes.get(ticker).get(test_run).get('training'))
                    scaler.transform(self.__test_runs_dataframes.get(ticker).get(test_run).get('training'))
                    scaler.transform(self.__test_runs_dataframes.get(ticker).get(test_run).get('validation'))
                    scaler.transform(self.__test_runs_dataframes.get(ticker).get(test_run).get('test'))
                    ticker_test_runs_scalers.update({test_run: scaler})
                    self.plot_data(f'{ticker} normalized data subset (test run {test_run})',
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('training'),
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('validation'),
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('test'),
                                   f'{png_export_path}/test_run_{test_run}_phase_2.png')
                self.__scalers.update({ticker: ticker_test_runs_scalers})

    def denoise_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                png_export_path = f'./images/data-preprocessing/{ticker}'
                if not os.path.exists(png_export_path):
                    os.makedirs(png_export_path)
                for test_run in self.__test_runs_periods.keys():
                    for data_set in ['training', 'validation', 'test']:
                        dataframe_index = self.__test_runs_dataframes.get(ticker).get(test_run).get(data_set).index
                        dataframe_values = (self.__test_runs_dataframes.get(ticker).get(test_run).get(data_set)
                                            .to_numpy(copy=True))
                        flattened_dataframe_values = dataframe_values.ravel()
                        denoised_values = tsd(flattened_dataframe_values)
                        if denoised_values.shape[0] > dataframe_index.shape[0]:
                            denoised_values = denoised_values[:-1]
                        self.__test_runs_dataframes.get(ticker).get(test_run).update(
                            {data_set: pd.DataFrame(denoised_values, index=dataframe_index, columns=['adj_close'])})
                    self.plot_data(f'{ticker} denoised data subset (test run {test_run})',
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('training'),
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('validation'),
                                   self.__test_runs_dataframes.get(ticker).get(test_run).get('test'),
                                   f'{png_export_path}/test_run_{test_run}_phase_3.png')

    def export_dataframes(self) -> None:
        model_data_path = f'./data'
        if not os.path.exists(model_data_path):
            os.makedirs(model_data_path)
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                ticker_data_path = f'{model_data_path}/{ticker}'
                if not os.path.exists(ticker_data_path):
                    os.makedirs(ticker_data_path)
                for test_run in self.__test_runs_periods.keys():
                    self.__test_runs_dataframes.get(ticker).get(test_run).get('training').to_csv(
                        f'{ticker_data_path}/test_run_{test_run}_training.csv')
                    self.__test_runs_dataframes.get(ticker).get(test_run).get('validation').to_csv(
                        f'{ticker_data_path}/test_run_{test_run}_validation.csv')
                    self.__test_runs_dataframes.get(ticker).get(test_run).get('test').to_csv(
                        f'{ticker_data_path}/test_run_{test_run}_test.csv')
                    dump(self.__scalers.get(ticker).get(test_run),
                         f'{ticker_data_path}/test_run_{test_run}_scaler.joblib')
                self.__cached_tickers.update({ticker: True})
                print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] Ticker {ticker} data downloaded, preprocessed '
                      f'and locally saved. {Style.RESET_ALL}')

    def init_datasets(self) -> None:
        for ticker in self.__tickers:
            ticker_test_run_datasets = {}
            for test_run in self.__test_runs_periods.keys():
                training_set_inputs, training_set_targets = self.df_to_timeseries(self.__test_runs_dataframes
                                                                                  .get(ticker).get(test_run)
                                                                                  .get('training'))
                validation_set_inputs, validation_set_targets = self.df_to_timeseries(self.__test_runs_dataframes
                                                                                      .get(ticker).get(test_run)
                                                                                      .get('validation'))
                test_set_inputs, test_set_targets = self.df_to_timeseries(self.__test_runs_dataframes.get(ticker)
                                                                          .get(test_run).get('test'))
                ticker_test_run_datasets.update({
                    test_run: {
                        'training': {
                            'inputs': training_set_inputs,
                            'targets': training_set_targets
                        },
                        'validation': {
                            'inputs': validation_set_inputs,
                            'targets': validation_set_targets
                        },
                        'test': {
                            'inputs': test_set_inputs,
                            'targets': test_set_targets
                        }
                    }
                })
            self.__datasets.update({ticker: ticker_test_run_datasets})
