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

    def plot_dataset(self, ticker, trading_day, shift, metric, training_data, validation_data, test_data, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} - {trading_day} - shift nr.{shift} - freq. {metric}')
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
            dataframe = pd.read_csv(ticker_path, sep=',')
            dataframe['TRADE_TIMESTAMP'] = pd.to_datetime(
                pd.to_datetime(dataframe['TRADE_DATE'], format='%d/%m/%Y').astype(str) + ' ' +
                dataframe['TRADE_TIME'].astype(str), format='%Y-%m-%d %H%M%S%f')
            dataframe.set_index('TRADE_TIMESTAMP', inplace=True)
            dataframe.drop(['ISIN', 'NAME', 'TRADE_DATE', 'TRADE_TIME', 'TRADESIZE', 'TURNOVER'], axis=1,
                           inplace=True)
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
                if trading_day.to_date_string() in self.__dataframes.get(ticker).index:
                    for shift in range(0, self.__config.dly_tr_amt):
                        begin, end = self.__config.get_shifts_schedules(trading_day, shift)
                        shift_dataframe = self.__dataframes.get(ticker).copy()[begin:end]
                        shift_dataframe = shift_dataframe.groupby(shift_dataframe.index).agg({'PRICE': 'mean'})
                        training_split, validation_split = self.__config.get_shifts_splits(shift_dataframe.shape[0])
                        training_dataframe = shift_dataframe[:training_split]
                        validation_dataframe = shift_dataframe[training_split:validation_split]
                        test_dataframe = shift_dataframe[validation_split:]
                        for resample in ['1s', '5s', '10s']:
                            training_dataframe_resampled = training_dataframe.resample(resample).mean().ffill()
                            validation_dataframe_resampled = validation_dataframe.resample(resample).mean().ffill()
                            test_dataframe_resampled = test_dataframe.resample(resample).mean().ffill()
                            self.plot_dataset(ticker, trading_day.format('dddd Do [of] MMMM YYYY'), shift, resample,
                                              training_dataframe_resampled, validation_dataframe_resampled,
                                              test_dataframe_resampled, f'{ticker_images_path}/'
                                                                        f'{trading_day.to_date_string()}_shift_{shift}'
                                                                        f'_{resample}.png')
                            training_dataframe_resampled.to_csv(f'{ticker_datasets_path}/{trading_day.to_date_string()}'
                                                                f'_shift_{shift}_training_{resample}.csv')
                            validation_dataframe_resampled.to_csv(f'{ticker_datasets_path}/'
                                                                  f'{trading_day.to_date_string()}_shift_{shift}'
                                                                  f'_validation_{resample}.csv')
                            test_dataframe_resampled.to_csv(f'{ticker_datasets_path}/{trading_day.to_date_string()}'
                                                            f'_shift_{shift}_test_{resample}.csv')
                    exported_test_runs += 1
            exported_tickers += 1
