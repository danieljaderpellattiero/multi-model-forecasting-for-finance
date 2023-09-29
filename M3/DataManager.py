import os
import matplotlib
import numpy as np
import pandas as pd
import pendulum as time
import urllib3 as request
import urllib3.exceptions
import yfinance as finance
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

matplotlib.use('Agg')


class DataManager:

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

    @staticmethod
    def plot_data(title, training, validation, test, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.plot(training, label='training_set')
        plt.plot(validation, label='validation_set')
        plt.plot(test, label='test_set')
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig(f'{path}')
        plt.close()

    @staticmethod
    def df_to_timeseries(dataframe, windows_size) -> tuple:
        timeseries = keras.utils.timeseries_dataset_from_array(
            dataframe.values.ravel(),
            None,
            sequence_length=windows_size + 1,
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
            validation_end = training_end.add(months=self.__config.tr_delay_m)
            period_end = validation_end.add(months=self.__config.tr_delay_m)
            self.__test_runs_periods.update({test_run: [period_begin, training_end, validation_end, period_end]})

    def import_local_data(self) -> None:
        data_path = f'./data'
        if os.path.exists(data_path):
            imported_dataframes = 0
            for ticker in self.__tickers:
                ticker_path = f'{data_path}/{ticker}'
                if os.path.exists(ticker_path):
                    is_missing_data = False
                    dataframes = {}
                    processed_dataframes = {}
                    for test_run in self.__test_runs_periods.keys():
                        if not is_missing_data:
                            test_run_path = f'{ticker_path}/test_run_{test_run}'
                            training_set_path = f'{test_run_path}/training.csv'
                            training_proc_set_path = f'{test_run_path}/training_processed.csv'
                            validation_set_path = f'{test_run_path}/validation.csv'
                            validation_proc_set_path = f'{test_run_path}/validation_processed.csv'
                            test_set_path = f'{test_run_path}/test.csv'
                            test_proc_set_path = f'{test_run_path}/test_processed.csv'
                            if (os.path.exists(test_run_path) and os.path.exists(training_set_path) and
                                    os.path.exists(training_proc_set_path) and os.path.exists(validation_set_path) and
                                    os.path.exists(validation_proc_set_path) and os.path.exists(test_set_path) and
                                    os.path.exists(test_proc_set_path)):
                                dataframes.update({
                                    test_run: {
                                        'training': pd.read_csv(training_set_path, index_col='Date', parse_dates=True),
                                        'validation': pd.read_csv(validation_set_path, index_col='Date',
                                                                  parse_dates=True),
                                        'test': pd.read_csv(test_set_path, index_col='Date', parse_dates=True)
                                    }
                                })
                                processed_dataframes.update({
                                    test_run: {
                                        'training': pd.read_csv(training_proc_set_path, index_col='Date',
                                                                parse_dates=True),
                                        'validation': pd.read_csv(validation_proc_set_path, index_col='Date',
                                                                  parse_dates=True),
                                        'test': pd.read_csv(test_proc_set_path, index_col='Date', parse_dates=True)
                                    }
                                })
                            else:
                                is_missing_data = True
                    if not is_missing_data:
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

    def is_local_data_available(self) -> bool:
        if (len(self.__test_runs_dataframes) == len(self.__tickers) and
                len(self.__test_runs_processed_dataframes) == len(self.__tickers)):
            return True
        else:
            return False

    def download_dataframes(self) -> None:
        if self.check_internet_connection():
            for ticker in self.__tickers:
                if not self.__cached_tickers.get(ticker):
                    png_export = f'./images/data-preprocessing/{ticker}'
                    if not os.path.exists(png_export):
                        os.makedirs(png_export)

                    dataframe = finance.download(ticker, self.__test_runs_periods.get(0)[0],
                                                 self.__test_runs_periods.get(self.__config.tr_amt - 1)[3].add(days=1),
                                                 progress=False)
                    dataframe = dataframe[['Adj Close']]
                    dataframe = dataframe.dropna(subset=['Adj Close'], inplace=False)
                    dataframe = dataframe.rename(columns={'Adj Close': 'adj_close'}, inplace=False)

                    dataframes = {}
                    for test_run in self.__test_runs_periods.keys():
                        dataframe_tmp = dataframe.copy(deep=True)
                        periods = self.__test_runs_periods.get(test_run)
                        dataframes.update({
                            test_run: {
                                'training': dataframe_tmp.loc[f'{periods[0].to_date_string()}':
                                                              f'{periods[1].to_date_string()}'],
                                'validation': dataframe_tmp.loc[f'{periods[1].add(days=1).to_date_string()}':
                                                                f'{periods[2].to_date_string()}'],
                                'test': dataframe_tmp.loc[f'{periods[2].add(days=1).to_date_string()}':
                                                          f'{periods[3].to_date_string()}']
                            }
                        })
                        self.plot_data(f'{ticker} original data subset (test run {test_run})',
                                       dataframes.get(test_run).get('training'),
                                       dataframes.get(test_run).get('validation'),
                                       dataframes.get(test_run).get('test'),
                                       f'{png_export}/test_run_{test_run}_phase_1.png')
                    self.__test_runs_dataframes.update({ticker: dataframes})
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] '
                  f'Cannot download dataframes without internet connection. {Style.RESET_ALL}')

    def homogenize_dataframes(self) -> None:
        for ticker in self.__tickers:
            png_export = f'./images/data-preprocessing/{ticker}'
            if not os.path.exists(png_export):
                os.makedirs(png_export)

            processed_dataframes = {}
            for test_run in self.__test_runs_periods.keys():
                processed_dataframes.update({
                    test_run: {
                        'training': np.log(self.__test_runs_dataframes.get(ticker).get(test_run).get('training')
                                           ['adj_close']),
                        'validation': np.log(self.__test_runs_dataframes.get(ticker).get(test_run).get('validation')
                                             ['adj_close']),
                        'test': np.log(self.__test_runs_dataframes.get(ticker).get(test_run).get('test')
                                       ['adj_close'])},
                })
                self.plot_data(f'{ticker} homogenized data subset (test run {test_run})',
                               processed_dataframes.get(test_run).get('training'),
                               processed_dataframes.get(test_run).get('validation'),
                               processed_dataframes.get(test_run).get('test'),
                               f'{png_export}/test_run_{test_run}_phase_2.png')
            self.__test_runs_processed_dataframes.update({ticker: processed_dataframes})

    def export_dataframes(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                data_path = f'./data/{ticker}'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                for test_run in self.__test_runs_periods.keys():
                    data_child_path = f'{data_path}/test_run_{test_run}'
                    if not os.path.exists(data_child_path):
                        os.makedirs(data_child_path)

                    self.__test_runs_dataframes.get(ticker).get(test_run).get('training').to_csv(
                        f'{data_child_path}/training.csv')
                    self.__test_runs_dataframes.get(ticker).get(test_run).get('validation').to_csv(
                        f'{data_child_path}/validation.csv')
                    self.__test_runs_dataframes.get(ticker).get(test_run).get('test').to_csv(
                        f'{data_child_path}/test.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('training').to_csv(
                        f'{data_child_path}/training_processed.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('validation').to_csv(
                        f'{data_child_path}/validation_processed.csv')
                    self.__test_runs_processed_dataframes.get(ticker).get(test_run).get('test').to_csv(
                        f'{data_child_path}/test_processed.csv')
                self.__cached_tickers.update({ticker: True})

    def init_datasets(self) -> None:
        for ticker in self.__tickers:
            datasets = {}
            for test_run in self.__test_runs_periods.keys():
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
            for test_run in self.__test_runs_periods.keys():
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
            for test_run in self.__test_runs_periods.keys():
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

    def reconstruct_and_export_results(self, ticker) -> None:
        for test_run in self.__test_runs_periods.keys():
            png_path = f'./images/predictions/{ticker}'
            if not os.path.exists(png_path):
                os.makedirs(png_path)

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
            predictions_rmse = np.sqrt(predictions_mse)
            predictions = np.exp(predictions)
            backtracked_predictions_aes = self.calculate_backtrack_aes(
                self.__test_runs_backtracked_predictions.get(ticker).get(test_run))
            self.export_results(ticker, test_run, predictions, predictions_mae, predictions_mape,
                                predictions_mse, predictions_rmse, backtracked_predictions_aes)
            self.plot_results(ticker, test_run, predictions, png_path)

    def export_results(self, ticker, test_run, predictions, predictions_mae, predictions_mape, predictions_mse,
                       predictions_rmse, backtracked_aes) -> None:
        data_path = f'./predictions/{ticker}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        model_results = pd.DataFrame({
            'forecasted_values': predictions.flatten(),
            'backtracked_values_aes': backtracked_aes.flatten(),
            'mae': predictions_mae,
            'mape': predictions_mape,
            'mse': predictions_mse,
            'rmse': predictions_rmse
        }, index=self.__test_runs_dataframes.get(ticker).get(test_run).get('test').index[self.__config.window_size:])
        model_results.to_csv(f'{data_path}/test_run_{test_run}.csv', encoding='utf-8',
                             sep=',', decimal='.', index_label='Date')

    def plot_results(self, ticker, test_run, predictions, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} forecasting results (test run {test_run})')
        plt.plot(pd.concat([
            self.__test_runs_dataframes.get(ticker).get(test_run).get('training'),
            self.__test_runs_dataframes.get(ticker).get(test_run).get('validation'),
            self.__test_runs_dataframes.get(ticker).get(test_run).get('test')
        ], axis=0), label='adj_close')
        plt.plot(pd.Series(
            predictions.flatten(),
            index=self.__test_runs_dataframes.get(ticker).get(test_run).get('test').index[self.__config.window_size:]
        ), label='model_predictions')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{path}/test_run_{test_run}.png')
        plt.close()
