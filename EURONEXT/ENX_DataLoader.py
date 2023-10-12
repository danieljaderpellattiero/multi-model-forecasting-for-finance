import os
import matplotlib
import pandas as pd

from Config import Config
from colorama import Fore, Style
from matplotlib import pyplot as plt

matplotlib.use('Agg')


class EuronextDataLoader:
    data_path = './data'
    images_path = './images'
    datasets_path = './datasets'
    data_plot_colors = ['royalblue', 'goldenrod', 'coral']

    def __init__(self, config_params, tickers) -> None:
        self.__uuid = 'ENX_DL'
        self.__tickers = tickers
        self.__config = Config(self.__uuid, config_params)
        self.__dataframes = {}

    def plot_dataset(self, ticker, trading_day, test_run, metric, training_data, validation_data, test_data, path) \
            -> None:
        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} - {trading_day} - test run {test_run} - freq. {metric}')
        plt.plot(training_data, label='training_set', color=self.data_plot_colors[0])
        plt.plot(validation_data, label='validation_set', color=self.data_plot_colors[1])
        plt.plot(test_data, label='test_set', color=self.data_plot_colors[2])
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f'{path}')
        plt.close()

    def load_data(self, tickers_limit) -> None:
        if tickers_limit > len(self.__tickers) or tickers_limit < 1:
            print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                  f'The specified limit exceeds the data availability or is malformed.')
            exit(1)
        loaded_tickers = 0
        for ticker in self.__tickers:
            if loaded_tickers >= tickers_limit:
                break
            ticker_path = f'{self.data_path}/{ticker}.csv'
            if not os.path.exists(ticker_path):
                print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] {ticker} trading data not found. {Style.RESET_ALL}')
                exit(1)
            dataframe = pd.read_csv(ticker_path, sep=',', header=0, index_col=False)
            dataframe['Trade_timestamp'] = pd.to_datetime(
                pd.to_datetime(dataframe['TRADE_DATE'], format='%d/%m/%Y').astype(str) + ' ' +
                dataframe['TRADE_TIME'].astype(str), format='%Y-%m-%d %H%M%S%f')
            dataframe.set_index('Trade_timestamp', inplace=True)
            dataframe.drop(['ISIN', 'NAME', 'TRADE_DATE', 'TRADE_TIME', 'TRADESIZE', 'TURNOVER'], axis=1,
                           inplace=True)
            dataframe.rename(columns={'PRICE': 'trade_price'}, inplace=True)
            self.__dataframes.update({ticker: dataframe})
            loaded_tickers += 1

    def export_datasets(self, tickers_limit, test_runs_limit) -> None:
        if (tickers_limit > len(self.__tickers) or tickers_limit < 1 or
                test_runs_limit > self.__config.duration or test_runs_limit < 1):
            print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                  f'The specified limits exceed the data availability or are malformed.')
            exit(1)
        exported_tickers = 0
        for ticker in self.__tickers:
            if exported_tickers >= tickers_limit:
                break
            ticker_images_path = f'{self.images_path}/{ticker}'
            ticker_datasets_path = f'{self.datasets_path}/{ticker}'
            if not os.path.exists(ticker_datasets_path):
                os.makedirs(ticker_datasets_path)
            if not os.path.exists(ticker_images_path):
                os.makedirs(ticker_images_path)
            exported_test_runs = 0
            for trading_day in self.__config.trading_days:
                if exported_test_runs >= test_runs_limit:
                    break
                daily_datasets_path = f'{ticker_datasets_path}/{trading_day.to_date_string()}'
                if not os.path.exists(daily_datasets_path):
                    os.makedirs(daily_datasets_path)
                if trading_day.to_date_string() in self.__dataframes.get(ticker).index:
                    for test_run in range(0, self.__config.dly_tr_amt):
                        datasets_path = f'{daily_datasets_path}/test_run_{test_run}'
                        if not os.path.exists(datasets_path):
                            os.makedirs(datasets_path)
                        begin, end = self.__config.get_test_run_schedules(trading_day, test_run)
                        dataframe = self.__dataframes.get(ticker).copy()[begin:end]
                        dataframe = dataframe.groupby(dataframe.index).agg({'trade_price': 'mean'})
                        for resample in ['1s']:
                            dataframe_resampled = dataframe.resample(resample).mean().ffill()
                            training_split, validation_split = self.__config.get_test_run_splits(dataframe_resampled
                                                                                                 .shape[0])
                            training_dataframe = dataframe_resampled[:training_split]
                            validation_dataframe = dataframe_resampled[training_split:validation_split]
                            test_dataframe = dataframe_resampled[validation_split:]
                            self.plot_dataset(ticker, trading_day.format('dddd Do [of] MMMM YYYY'), test_run, resample,
                                              training_dataframe, validation_dataframe, test_dataframe,
                                              f'{ticker_images_path}/{trading_day.to_date_string()}'
                                              f'_test_run_{test_run}_{resample}.png')
                            dataframe_resampled.to_csv(f'{datasets_path}/trade_price_{resample}.csv',
                                                       encoding='utf-8', sep=';', decimal=',')
                            training_dataframe.to_csv(f'{datasets_path}/training_{resample}.csv',
                                                      encoding='utf-8', sep=';', decimal=',')
                            validation_dataframe.to_csv(f'{datasets_path}/validation_{resample}.csv',
                                                        encoding='utf-8', sep=';', decimal=',')
                            test_dataframe.to_csv(f'{datasets_path}/test_{resample}.csv',
                                                  encoding='utf-8', sep=';', decimal=',')
                    exported_test_runs += 1
            exported_tickers += 1
