import os
import matplotlib
import numpy as np
import pandas as pd
import pendulum as time
import tensorflow as tf
import urllib3 as request
import urllib3.exceptions
import yfinance as finance
import matplotlib.pyplot as plt

from tensorflow import keras
from joblib import dump, load
from colorama import Fore, Style
from Wavelet_denoising import tsd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

matplotlib.use('Agg')


class DataManager:
    data_path = './data'
    enx_data_path = './enx_data'
    images_path = './images'
    predictions_path = './predictions'
    data_plot_colors = ['royalblue', 'goldenrod', 'coral']
    model_predictions_plot_color = 'orchid'

    def __init__(self, model_config, tickers) -> None:
        self.__tickers = tickers
        self.__config = model_config
        self.__cached_tickers = {}
        self.__test_runs_periods = {}
        self.__test_runs_scalers = {}
        self.__test_runs_dataframes = {}
        self.__test_runs_processed_dataframes = {}
        self.__test_runs_datasets = {}
        self.__test_runs_batched_datasets = {}
        self.__test_runs_backtracked_datasets = {}
        self.__test_runs_predictions = {}
        self.__test_runs_backtracked_predictions = {}

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
    def tr_datasets(self) -> dict:
        return self.__test_runs_datasets

    @property
    def tr_btch_datasets(self) -> dict:
        return self.__test_runs_batched_datasets

    @property
    def tr_bt_datasets(self) -> dict:
        return self.__test_runs_backtracked_datasets

    @property
    def tr_predictions(self) -> dict:
        return self.__test_runs_predictions

    @property
    def tr_bt_predictions(self) -> dict:
        return self.__test_runs_backtracked_predictions

    @staticmethod
    def check_internet_connection() -> bool:
        try:
            response = (request.connection_from_url('https://www.google.com')
                        .request('GET', '/', timeout=1.0))
            return True if response.status == 200 else False
        except urllib3.exceptions.NewConnectionError:
            return False
        except urllib3.exceptions.MaxRetryError:
            return False

    def plot_dataset(self, title, training_data, validation_data, test_data, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.plot(training_data, label='training_set', color=self.data_plot_colors[0])
        plt.plot(validation_data, label='validation_set', color=self.data_plot_colors[1])
        plt.plot(test_data, label='test_set', color=self.data_plot_colors[2])
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f'{path}')
        plt.close()

    def plot_predictions(self, ticker, test_run, predictions, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} forecasting outcome - test run {test_run}')
        plt.plot(self.__test_runs_dataframes.get(ticker).get(test_run).tail(predictions.shape[0]),
                 label='adj_close' if not self.__config.enx_data else 'trade_price',
                 color=self.data_plot_colors[0])
        plt.plot(pd.Series(
            predictions.flatten(),
            index=self.__test_runs_dataframes.get(ticker).get(test_run).tail(predictions.shape[0]).index
        ), label='model_predictions', color=self.model_predictions_plot_color)
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{path}/test_run_{test_run}.png')
        plt.close()

    @staticmethod
    def df_to_timeseries(dataframe, window_size) -> tuple:
        timeseries = keras.utils.timeseries_dataset_from_array(
            dataframe.values.ravel(),
            None,
            sequence_length=window_size + 1,
            sequence_stride=1,
            sampling_rate=1,
            shuffle=False
        )
        dataset = list(timeseries.as_numpy_iterator())
        dataset = np.concatenate(dataset, axis=0)
        return dataset[:, :-1], dataset[:, -1].reshape(-1, 1)

    @staticmethod
    def search_closest_sequence(target_sequence, training_set_sequences, validation_set_sequences) -> np.ndarray:
        best_match = None
        best_match_distance = None
        for sequence_index, sequence in enumerate(training_set_sequences):
            distance = np.linalg.norm(sequence - target_sequence)
            if best_match is None or distance < best_match_distance:
                best_match = sequence_index
                best_match_distance = distance
        for sequence_index, sequence in enumerate(validation_set_sequences):
            distance = np.linalg.norm(sequence - target_sequence)
            if best_match is None or distance < best_match_distance:
                best_match = training_set_sequences.shape[0] + sequence_index
                best_match_distance = distance
        return best_match

    @staticmethod
    def init_backtrack_buffer(target) -> np.ndarray:
        return np.zeros((target.shape[0], 2), dtype=float)

    @staticmethod
    def calculate_backtrack_aes(predictions) -> np.ndarray:
        metrics = np.zeros((predictions.shape[0], 1), dtype=float)
        for index in range(predictions.shape[0]):
            metrics[index] = np.abs(predictions[index][1] - predictions[index][0])
        return metrics

    def init_periods(self) -> None:
        period_begin = time.from_format(self.__config.period_begin, 'YYYY-MM-DD', tz='Europe/Rome')
        for test_run in range(0, self.__config.tr_amt):
            if test_run != 0:
                period_begin = period_begin.add(months=self.__config.tr_step_size)
            training_end = period_begin.add(years=self.__config.tr_delay_y)
            validation_end = training_end.add(months=self.__config.tr_delay_m[0])
            period_end = validation_end.add(months=self.__config.tr_delay_m[1])
            self.__test_runs_periods.update({test_run: [period_begin, training_end, validation_end, period_end]})

    def import_local_data(self) -> None:
        if os.path.exists(self.data_path):
            imported_dataframes = 0
            for ticker in self.__tickers:
                ticker_path = f'{self.data_path}/{ticker}'
                if os.path.exists(ticker_path):
                    is_missing_data = False
                    scalers = {}
                    dataframes = {}
                    processed_dataframes = {}
                    for test_run in range(0, self.__config.tr_amt):
                        if not is_missing_data:
                            test_run_path = f'{ticker_path}/test_run_{test_run}'
                            dataframe_path = (f'{test_run_path}/adj_close.csv' if not self.__config.enx_data else
                                              f'{test_run_path}/trade_price_{self.__config.enx_data_freq}.csv')
                            training_set_path = (f'{test_run_path}/training.csv' if not self.__config.enx_data else
                                                 f'{test_run_path}/training_{self.__config.enx_data_freq}.csv')
                            validation_set_path = (f'{test_run_path}/validation.csv' if not self.__config.enx_data else
                                                   f'{test_run_path}/validation_{self.__config.enx_data_freq}.csv')
                            test_set_path = (f'{test_run_path}/test.csv' if not self.__config.enx_data else
                                             f'{test_run_path}/test_{self.__config.enx_data_freq}.csv')
                            scaler_path = (f'{test_run_path}/scaler.joblib' if not self.__config.enx_data else
                                           f'{test_run_path}/scaler_{self.__config.enx_data_freq}.joblib')
                            if (os.path.exists(test_run_path) and os.path.exists(dataframe_path) and
                                    os.path.exists(training_set_path) and os.path.exists(validation_set_path) and
                                    os.path.exists(test_set_path) and os.path.exists(scaler_path)):
                                dataframes.update({test_run: pd.read_csv(dataframe_path,
                                                                         index_col=('Date' if not self.__config.enx_data
                                                                                    else 'Trade_timestamp'),
                                                                         parse_dates=True)})
                                processed_dataframes.update({
                                    test_run: {
                                        'training': pd.read_csv(training_set_path,
                                                                index_col=('Date' if not self.__config.enx_data
                                                                           else 'Trade_timestamp'), parse_dates=True),
                                        'validation': pd.read_csv(validation_set_path,
                                                                  index_col=('Date' if not self.__config.enx_data
                                                                             else 'Trade_timestamp'), parse_dates=True),
                                        'test': pd.read_csv(test_set_path,
                                                            index_col=('Date' if not self.__config.enx_data
                                                                       else 'Trade_timestamp'), parse_dates=True),
                                    }
                                })
                                scalers.update({test_run: load(scaler_path)})
                            else:
                                is_missing_data = True
                    if not is_missing_data:
                        self.__test_runs_scalers.update({ticker: scalers})
                        self.__test_runs_dataframes.update({ticker: dataframes})
                        self.__test_runs_processed_dataframes.update({ticker: processed_dataframes})
                        self.__cached_tickers.update({ticker: True})
                        imported_dataframes += 1
                        print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] Ticker {ticker} imported. '
                              f'{Style.RESET_ALL}')
                    else:
                        print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] '
                              f'Ticker {ticker} not imported due to missing data. {Style.RESET_ALL}')
                else:
                    print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] '
                          f'No local data found for ticker {ticker}. {Style.RESET_ALL}')
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] {imported_dataframes} out of {len(self.__tickers)} '
                  f'ticker(s) imported from local data. {Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] No local data found for the model. '
                  f'{Style.RESET_ALL}')

    def check_data_availability(self) -> bool:
        if (len(self.__test_runs_dataframes) == len(self.__tickers) and
                len(self.__test_runs_processed_dataframes) == len(self.__tickers)):
            return True
        else:
            return False

    def download_dataframes(self) -> None:
        if self.check_internet_connection():
            for ticker in self.__tickers:
                if not self.__cached_tickers.get(ticker):
                    ticker_png_path = f'{self.images_path}/data-preprocessing/{ticker}'
                    if not os.path.exists(ticker_png_path):
                        os.makedirs(ticker_png_path)
                    dataframe = finance.download(ticker, self.__test_runs_periods.get(0)[0],
                                                 self.__test_runs_periods.get(self.__config.tr_amt - 1)[3].add(days=1),
                                                 progress=False)
                    dataframe = dataframe[['Adj Close']]
                    dataframe = dataframe.dropna(subset=['Adj Close'], inplace=False)
                    dataframe = dataframe.rename(columns={'Adj Close': 'adj_close'}, inplace=False)
                    dataframes = {}
                    processed_dataframes = {}
                    for test_run in range(0, self.__config.tr_amt):
                        dataframe_tmp_1 = dataframe.copy(deep=True)
                        dataframe_tmp_2 = dataframe.copy(deep=True)
                        periods = self.__test_runs_periods.get(test_run)
                        dataframes.update({
                            test_run: dataframe_tmp_1.loc[f'{periods[0].to_date_string()}':
                                                          f'{periods[3].to_date_string()}']
                        })
                        processed_dataframes.update({
                            test_run: {
                                'training': dataframe_tmp_2.loc[f'{periods[0].to_date_string()}':
                                                                f'{periods[1].to_date_string()}'],
                                'validation': dataframe_tmp_2.loc[f'{periods[1].add(days=1).to_date_string()}':
                                                                  f'{periods[2].to_date_string()}'],
                                'test': dataframe_tmp_2.loc[f'{periods[2].add(days=1).to_date_string()}':
                                                            f'{periods[3].to_date_string()}']
                            }
                        })
                        self.plot_dataset(f'{ticker} original data subset - test run {test_run}',
                                          processed_dataframes.get(test_run).get('training'),
                                          processed_dataframes.get(test_run).get('validation'),
                                          processed_dataframes.get(test_run).get('test'),
                                          f'{ticker_png_path}/test_run_{test_run}_phase_1.png')
                    self.__test_runs_dataframes.update({ticker: dataframes})
                    self.__test_runs_processed_dataframes.update({ticker: processed_dataframes})
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] '
                  f'Cannot download dataframes without internet connection. {Style.RESET_ALL}')

    def import_enx_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                ticker_png_path = f'{self.images_path}/data-preprocessing/{ticker}'
                if not os.path.exists(ticker_png_path):
                    os.makedirs(ticker_png_path)
                dataframes = {}
                processed_dataframes = {}
                for test_run in range(0, self.__config.tr_amt):
                    test_run_path = f'{self.enx_data_path}/{ticker}/test_run_{test_run}'
                    dataframe_path = f'{test_run_path}/trade_price_{self.__config.enx_data_freq}.csv'
                    training_set_path = f'{test_run_path}/training_{self.__config.enx_data_freq}.csv'
                    validation_set_path = f'{test_run_path}/validation_{self.__config.enx_data_freq}.csv'
                    test_set_path = f'{test_run_path}/test_{self.__config.enx_data_freq}.csv'
                    if (os.path.exists(test_run_path) and os.path.exists(dataframe_path) and
                            os.path.exists(training_set_path) and os.path.exists(validation_set_path) and
                            os.path.exists(test_set_path)):
                        dataframes.update({test_run: pd.read_csv(dataframe_path, index_col='Trade_timestamp',
                                                                 parse_dates=True)})
                        processed_dataframes.update({
                            test_run: {
                                'training': pd.read_csv(training_set_path, index_col='Trade_timestamp',
                                                        parse_dates=True),
                                'validation': pd.read_csv(validation_set_path, index_col='Trade_timestamp',
                                                          parse_dates=True),
                                'test': pd.read_csv(test_set_path, index_col='Trade_timestamp', parse_dates=True)
                            }
                        })
                        self.plot_dataset(f'{ticker} original data subset - {test_run}',
                                          processed_dataframes.get(test_run).get('training'),
                                          processed_dataframes.get(test_run).get('validation'),
                                          processed_dataframes.get(test_run).get('test'),
                                          f'{ticker_png_path}/test_run_{test_run}_phase_1.png')
                    else:
                        print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] '
                              f'Euronext trading data not found for ticker {ticker} in test run {test_run}. '
                              f'{Style.RESET_ALL}')
                        exit(1)
                self.__test_runs_dataframes.update({ticker: dataframes})
                self.__test_runs_processed_dataframes.update({ticker: processed_dataframes})

    def normalize_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                ticker_png_path = f'{self.images_path}/data-preprocessing/{ticker}'
                if not os.path.exists(ticker_png_path):
                    os.makedirs(ticker_png_path)
                scalers = {}
                for test_run in range(0, self.__config.tr_amt):
                    scaler = MinMaxScaler(copy=False, clip=False)
                    scaler.fit(self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training'))
                    scaler.transform(self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training'))
                    scaler.transform(self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation'))
                    scaler.transform(self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test'))
                    scalers.update({test_run: scaler})
                    self.plot_dataset(f'{ticker} normalized data subset - test run {test_run}',
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training'),
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation'),
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test'),
                                      f'{ticker_png_path}/test_run_{test_run}_phase_2.png')
                self.__test_runs_scalers.update({ticker: scalers})

    def denoise_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                ticker_png_path = f'{self.images_path}/data-preprocessing/{ticker}'
                if not os.path.exists(ticker_png_path):
                    os.makedirs(ticker_png_path)
                for test_run in range(0, self.__config.tr_amt):
                    for data_set in ['training', 'validation', 'test']:
                        dataframe_index = (self.__test_runs_processed_dataframes.get(ticker).get(test_run)
                                           .get(data_set).index)
                        dataframe_values = (self.__test_runs_processed_dataframes.get(ticker).get(test_run)
                                            .get(data_set).to_numpy(copy=True))
                        flattened_dataframe_values = dataframe_values.ravel()
                        denoised_values = tsd(flattened_dataframe_values)
                        if denoised_values.shape[0] > dataframe_index.shape[0]:
                            denoised_values = denoised_values[:dataframe_index.shape[0]]
                        self.__test_runs_processed_dataframes.get(ticker).get(test_run).update({
                            data_set: pd.DataFrame(denoised_values, index=dataframe_index,
                                                   columns=[('adj_close' if not self.__config.enx_data
                                                             else 'trade_price')])
                        })
                    self.plot_dataset(f'{ticker} denoised data subset - test run {test_run}',
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training'),
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation'),
                                      self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test'),
                                      f'{ticker_png_path}/test_run_{test_run}_phase_3.png')

    def export_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                ticker_data_path = f'{self.data_path}/{ticker}'
                if not os.path.exists(ticker_data_path):
                    os.makedirs(ticker_data_path)
                for test_run in range(0, self.__config.tr_amt):
                    data_child_path = f'{ticker_data_path}/test_run_{test_run}'
                    if not os.path.exists(data_child_path):
                        os.makedirs(data_child_path)
                    self.__test_runs_dataframes.get(ticker).get(test_run).to_csv(
                        f'{data_child_path}/adj_close.csv' if not self.__config.enx_data else
                        f'{data_child_path}/trade_price_{self.__config.enx_data_freq}.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training').to_csv(
                        f'{data_child_path}/training.csv' if not self.__config.enx_data else
                        f'{data_child_path}/training_{self.__config.enx_data_freq}.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation').to_csv(
                        f'{data_child_path}/validation.csv' if not self.__config.enx_data else
                        f'{data_child_path}/validation_{self.__config.enx_data_freq}.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test').to_csv(
                        f'{data_child_path}/test.csv' if not self.__config.enx_data else
                        f'{data_child_path}/test_{self.__config.enx_data_freq}.csv')
                    dump(self.__test_runs_scalers.get(ticker).get(test_run),
                         f'{data_child_path}/scaler.joblib' if not self.__config.enx_data else
                         f'{data_child_path}/scaler_{self.__config.enx_data_freq}.joblib')
                self.__cached_tickers.update({ticker: True})

    def init_datasets(self) -> None:
        for ticker in self.__tickers:
            datasets = {}
            for test_run in range(0, self.__config.tr_amt):
                training_set_inputs, training_set_targets = self.df_to_timeseries(
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training'),
                    self.__config.window_size)
                validation_set_inputs, validation_set_targets = self.df_to_timeseries(
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation'),
                    self.__config.window_size)
                test_set_inputs, test_set_targets = self.df_to_timeseries(
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test'),
                    self.__config.window_size)
                datasets.update({
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
            self.__test_runs_datasets.update({ticker: datasets})

    def init_batches(self) -> None:
        for ticker in self.__tickers:
            datasets_batched = {}
            for test_run in range(0, self.__config.tr_amt):
                training_set = tf.data.Dataset.from_tensor_slices((
                    tf.convert_to_tensor(
                        self.__test_runs_datasets.get(ticker).get(test_run).get('training').get('inputs')),
                    tf.convert_to_tensor(
                        self.__test_runs_datasets.get(ticker).get(test_run).get('training').get('targets'))
                )).batch(self.__config.batch_size)
                validation_set = tf.data.Dataset.from_tensor_slices((
                    tf.convert_to_tensor(
                        self.__test_runs_datasets.get(ticker).get(test_run).get('validation').get('inputs')),
                    tf.convert_to_tensor(
                        self.__test_runs_datasets.get(ticker).get(test_run).get('validation').get('targets'))
                )).batch(self.__config.batch_size)
                datasets_batched.update({
                    test_run: {
                        'training': training_set,
                        'validation': validation_set,
                        'test': {
                            'inputs': self.__test_runs_datasets.get(ticker).get(test_run).get('test').get('inputs'),
                            'targets': self.__test_runs_datasets.get(ticker).get(test_run).get('test').get('targets')
                        }
                    }
                })
            self.__test_runs_batched_datasets.update({ticker: datasets_batched})

    def init_alternative_dataset(self) -> None:
        for ticker in self.__tickers:
            dataset = {}
            for test_run in range(0, self.__config.tr_amt):
                training_sequences = (self.__test_runs_datasets.get(ticker).get(test_run)
                                      .get('training').get('inputs'))
                validation_sequences = (self.__test_runs_datasets.get(ticker).get(test_run)
                                        .get('validation').get('inputs'))
                test_sequences = (self.__test_runs_datasets.get(ticker).get(test_run)
                                  .get('test').get('inputs'))
                closest_sequences_refs = np.zeros((test_sequences.shape[0], 1), dtype=int)
                for sequence_index, sequence in enumerate(test_sequences):
                    closest_sequences_refs[sequence_index] = self.search_closest_sequence(sequence, training_sequences,
                                                                                          validation_sequences)
                dataset.update({test_run: closest_sequences_refs})
            self.__test_runs_backtracked_datasets.update({ticker: dataset})

    def reconstruct_and_export_predictions(self, ticker) -> None:
        for test_run in range(0, self.__config.tr_amt):
            scaler = self.__test_runs_scalers.get(ticker).get(test_run)
            predictions_png_path = f'{self.images_path}/predictions/{ticker}'
            if not os.path.exists(predictions_png_path):
                os.makedirs(predictions_png_path)
            predictions = np.array(self.__test_runs_predictions.get(ticker).get(test_run)).reshape(-1, 1)
            predictions_mae = mean_absolute_error(
                self.__test_runs_datasets.get(ticker).get(test_run).get('test').get('targets'),
                predictions)
            predictions_mape = mean_absolute_percentage_error(
                self.__test_runs_datasets.get(ticker).get(test_run).get('test').get('targets'),
                predictions)
            predictions_mse = mean_squared_error(
                self.__test_runs_datasets.get(ticker).get(test_run).get('test').get('targets'),
                predictions)
            scaler.inverse_transform(predictions)
            backtracked_predictions_aes = self.calculate_backtrack_aes(
                self.__test_runs_backtracked_predictions.get(ticker).get(test_run))
            self.export_predictions(ticker, test_run, predictions, predictions_mae, predictions_mape,
                                    predictions_mse, backtracked_predictions_aes)
            self.plot_predictions(ticker, test_run, predictions, predictions_png_path)

    def export_predictions(self, ticker, test_run, predictions, predictions_mae, predictions_mape, predictions_mse,
                           backtracked_aes) -> None:
        predictions_path = f'{self.predictions_path}/{ticker}'
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        model_predictions = pd.DataFrame({
            'forecasted_values': predictions.flatten(),
            'backtracked_values_aes': backtracked_aes.flatten(),
            'mae': predictions_mae,
            'mape': predictions_mape,
            'mse': predictions_mse,
        }, index=self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test')
                 .index[-predictions.shape[0]:])
        model_predictions.to_csv((f'{predictions_path}/test_run_{test_run}.csv' if not self.__config.enx_data else
                                  f'{predictions_path}/test_run_{test_run}_{self.__config.enx_data_freq}.csv'),
                                 encoding='utf-8', sep=',', decimal=',',
                                 index_label='Date' if not self.__config.enx_data else 'Trade_timestamp')
