import os
import json
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
from TSDecomposer import TSDecomposer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


class DataManager:

    def __init__(self, model_config, tickers) -> None:
        self.__tickers = tickers
        self.__config = model_config
        self.__decomposer = TSDecomposer()
        self.__cached_tickers = {}
        self.__test_runs_periods = {}
        self.__test_runs_dataframes = {}
        self.__test_runs_components = {}
        self.__test_runs_components_scalers = {}
        self.__test_runs_components_learning_params = {}
        self.__test_runs_datasets = {}
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
    def tr_components(self) -> dict:
        return self.__test_runs_components

    @property
    def tr_learning_params(self) -> dict:
        return self.__test_runs_components_learning_params

    @property
    def tr_scalers(self) -> dict:
        return self.__test_runs_components_scalers

    @property
    def tr_datasets(self) -> dict:
        return self.__test_runs_datasets

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
            response = (request.connection_from_url('https://www.google.com').request('GET', '/',
                                                                                      timeout=1.0))
            return True if response.status == 200 else False
        except urllib3.exceptions.NewConnectionError:
            return False
        except urllib3.exceptions.MaxRetryError:
            return False

    @staticmethod
    def plot_time_series(title, time_series, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.plot(time_series, label='adj_close')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{path}')
        plt.close()

    # Utility method.
    @staticmethod
    def init_windows(target_windows) -> list or None:
        if target_windows < 2:
            return None
        windows = [-1] * target_windows
        usage_delta = 0.05
        windows_values = [2, 4]
        usage_upper_limit = 0.5
        usage_lower_limit = 0.2
        windows[0] = windows_values[0]
        windows[-1] = windows_values[1]
        if len(windows) > 2:
            usage_percentages = np.arange(usage_upper_limit, usage_lower_limit - usage_delta, -usage_delta).tolist()
            window_former_value_usage = (np.ceil(len(windows) *
                                                 usage_percentages[np.random.randint(0, len(usage_percentages))])
                                         .astype(int))
            for occurrence in range(1, window_former_value_usage):
                windows[occurrence] = windows_values[0]
            for remaining_occurrence in range(window_former_value_usage, len(windows)):
                windows[remaining_occurrence] = windows_values[1]
            return windows
        return windows

    @staticmethod
    def init_epochs(target_epochs) -> list or None:
        if target_epochs < 2:
            return None
        epochs: list[int] = [-1] * target_epochs
        delta = 10
        values_lower_limits = [250, 200]
        values_upper_limits = [300, 290]
        variable_delta = np.random.randint((values_lower_limits[0] - values_lower_limits[1]) // delta + 1) * delta
        epochs[0] = values_upper_limits[0]
        epochs[-1] = values_lower_limits[1] + variable_delta
        if len(epochs) > 2:
            remaining_epochs = len(epochs[1:-1])
            epochs_pool = np.arange(values_upper_limits[1], values_lower_limits[0] - delta, -delta).tolist()
            if remaining_epochs > len(epochs_pool):
                return None
            filler_epoch_index = 1
            pool_upper_limit = len(epochs_pool) - remaining_epochs + 1
            for pool_lower_limit in range(remaining_epochs):
                candidate_value = epochs_pool[np.random.randint(pool_lower_limit, pool_lower_limit + pool_upper_limit)]
                while candidate_value > epochs[filler_epoch_index - 1]:
                    candidate_value = int(epochs_pool[np.random.randint(pool_lower_limit,
                                                                        pool_lower_limit + pool_upper_limit)])
                epochs[filler_epoch_index] = candidate_value
                filler_epoch_index += 1
            return epochs
        return epochs

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
        for index, sequence in enumerate(training_set_sequences):
            distance = np.linalg.norm(sequence - target_sequence)
            if best_match is None or distance < best_match_distance:
                best_match = index
                best_match_distance = distance
        for index, sequence in enumerate(validation_set_sequences):
            distance = np.linalg.norm(sequence - target_sequence)
            if best_match is None or distance < best_match_distance:
                best_match = training_set_sequences.shape[0] + index
                best_match_distance = distance
        return best_match

    @staticmethod
    def init_backtrack_buffer(target) -> np.ndarray:
        return np.zeros((target.shape[0], 2), dtype=float)

    @staticmethod
    def calculate_backtrack_aes(components_predictions) -> np.ndarray:
        metrics = np.zeros((components_predictions.get(list(components_predictions.keys())[0]).shape[0], 1),
                           dtype=float)
        for index in enumerate(metrics):
            components_metrics = np.zeros((len(components_predictions.keys()),), dtype=float)
            for sub_index, component in enumerate(components_predictions.keys()):
                components_metrics[sub_index] = np.abs(components_predictions.get(component)[index][1] -
                                                       components_predictions.get(component)[index][0])
            metrics[index] = np.mean(components_metrics)
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
        data_path = './data'
        if os.path.exists(data_path):
            imported_tickers = 0
            for ticker in self.__tickers:
                ticker_path = f'{data_path}/{ticker}'
                if os.path.exists(ticker_path):
                    is_data_missing = False
                    dataframes = {}
                    time_series_components = {}
                    time_series_components_scalers = {}
                    time_series_components_learning_params = {}
                    for test_run in self.__test_runs_periods.keys():
                        if not is_data_missing:
                            test_run_path = f'{ticker_path}/test_run_{test_run}'
                            dataframe_path = f'{test_run_path}/adj_close.csv'
                            learning_params_path = f'{test_run_path}/learning_params.json'
                            if (os.path.exists(test_run_path) and os.path.exists(dataframe_path) and
                                    os.path.exists(learning_params_path)):
                                dataframes.update({test_run: pd.read_csv(dataframe_path, index_col='Date',
                                                                         parse_dates=True)})
                                with open(learning_params_path, 'r') as file:
                                    time_series_components_learning_params.update({test_run: json.load(file)})
                                time_series_components_tmp = {}
                                time_series_components_scalers_tmp = {}
                                for component in time_series_components_learning_params.get(test_run).get('components'):
                                    if not is_data_missing:
                                        component_path = f'{test_run_path}/{component}.csv'
                                        component_scaler_path = f'{test_run_path}/{component}_scaler.joblib'
                                        if os.path.exists(component_path) and os.path.exists(component_scaler_path):
                                            time_series_components_tmp.update({
                                                component: pd.read_csv(component_path, index_col='Date',
                                                                       parse_dates=True)
                                            })
                                            time_series_components_scalers_tmp.update({
                                                component: load(component_scaler_path)
                                            })
                                        else:
                                            is_data_missing = True
                                if not is_data_missing:
                                    time_series_components.update({
                                        test_run: time_series_components_tmp})
                                    time_series_components_scalers.update({
                                        test_run: time_series_components_scalers_tmp})
                            else:
                                is_data_missing = True
                    if not is_data_missing:
                        self.__test_runs_dataframes.update({ticker: dataframes})
                        self.__test_runs_components.update({ticker: time_series_components})
                        self.__test_runs_components_scalers.update({ticker: time_series_components_scalers})
                        self.__test_runs_components_learning_params.update({ticker:
                                                                            time_series_components_learning_params})
                        self.__cached_tickers.update({ticker: True})
                        imported_tickers += 1
                        print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] Ticker {ticker} imported. '
                              f'{Style.RESET_ALL}')
                    else:
                        print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] '
                              f'Ticker {ticker} not imported due to missing data. {Style.RESET_ALL}')
                else:
                    print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] No local data found for ticker {ticker}. '
                          f'{Style.RESET_ALL}')
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} ] '
                  f'{imported_tickers} out of {len(self.__tickers)} ticker(s) imported from local data. '
                  f'{Style.RESET_ALL}')
        else:
            print(f'{Fore.LIGHTYELLOW_EX} [ {self.__config.uuid} ] No local data found for the model. '
                  f'{Style.RESET_ALL}')

    def check_local_data_availability(self) -> bool:
        if (len(self.__test_runs_dataframes.keys()) == len(self.__tickers) and
                len(self.__test_runs_components.keys()) == len(self.__tickers) and
                len(self.__test_runs_components_scalers.keys()) == len(self.__tickers) and
                len(self.__test_runs_components_learning_params.keys()) == len(self.__tickers)):
            return True
        else:
            return False

    def download_dataframes(self) -> None:
        if self.check_internet_connection():
            for ticker in self.__tickers:
                if not self.__cached_tickers.get(ticker):
                    png_path = f'./images/data-preprocessing/{ticker}'
                    if not os.path.exists(png_path):
                        os.makedirs(png_path)

                    dataframe = finance.download(ticker,
                                                 self.__test_runs_periods.get(0)[0],
                                                 self.__test_runs_periods.get(self.__config.tr_amt - 1)[3].add(days=1),
                                                 progress=False)
                    dataframe = dataframe[['Adj Close']]
                    dataframe = dataframe.rename(columns={'Adj Close': 'adj_close'}, inplace=False)

                    dataframes = {}
                    for test_run in self.__test_runs_periods.keys():
                        dataframe_tmp = dataframe.copy(deep=True)
                        periods = self.__test_runs_periods.get(test_run)
                        dataframes.update({
                            test_run: dataframe_tmp.loc[f'{periods[0].to_date_string()}':
                                                        f'{periods[3].to_date_string()}']
                        })
                        self.plot_time_series(f'{ticker} original data subset (test run {test_run})',
                                              dataframes.get(test_run),
                                              f'{png_path}/test_run_{test_run}.png')
                    self.__test_runs_dataframes.update({ticker: dataframes})
        else:
            print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] '
                  f'Cannot download dataframes without internet connection. {Style.RESET_ALL}')

    def decompose_time_series(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                time_series_components = {}
                for test_run in self.__test_runs_periods.keys():
                    time_series_components.update({
                        test_run: self.__decomposer.decompose(self.__test_runs_dataframes.get(ticker).get(test_run),
                                                              ticker, test_run)
                    })
                    print(f'{Fore.LIGHTGREEN_EX} [ {self.__config.uuid} | {ticker} | Test run {test_run} ] '
                          f'Time series decomposed. {Style.RESET_ALL}')
                self.__test_runs_components.update({ticker: time_series_components})

    def normalize_time_series_components(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                png_path = f'./images/data-preprocessing/{ticker}'
                if not os.path.exists(png_path):
                    os.makedirs(png_path)

                time_series_scalers = {}
                for test_run in self.__test_runs_periods.keys():
                    time_series_component_scalers = {}
                    for component in self.__test_runs_components.get(ticker).get(test_run).keys():
                        scaler = MinMaxScaler(copy=False, clip=False)
                        scaler.fit_transform(self.__test_runs_components.get(ticker).get(test_run).get(component))
                        time_series_component_scalers.update({component: scaler})
                    time_series_scalers.update({test_run: time_series_component_scalers})
                    self.__decomposer.plot_components(
                        list(self.__test_runs_components.get(ticker).get(test_run).values())[:-1],
                        list(self.__test_runs_components.get(ticker).get(test_run).values())[-1],
                        self.__test_runs_dataframes.get(ticker).get(test_run).index,
                        ticker, test_run, 2)
                self.__test_runs_components_scalers.update({ticker: time_series_scalers})

    def init_learning_params(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                learning_params = {}
                for test_run in self.__test_runs_periods.keys():
                    components_amount = len(self.__test_runs_components.get(ticker).get(test_run))
                    learning_params.update({
                        test_run: {
                            'components': list(self.__test_runs_components.get(ticker).get(test_run).keys()),
                            'windows_size': [self.__config.window_size] * components_amount,
                            'epochs': self.init_epochs(components_amount)
                        }
                    })
                    if (learning_params.get(test_run).get('windows_size') is None or
                            learning_params.get(test_run).get('epochs') is None):
                        print(f'{Fore.LIGHTRED_EX} [ {self.__config.uuid} ] Cannot initialize learning parameters. '
                              f'{Style.RESET_ALL}')
                        quit(1)
                self.__test_runs_components_learning_params.update({ticker: learning_params})

    def export_time_series_components(self) -> None:
        for ticker in self.__tickers:
            if not self.__cached_tickers.get(ticker):
                data_path = f'./data/{ticker}'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                time_series_dataframes = {}
                for test_run in self.__test_runs_periods.keys():
                    data_child_path = f'{data_path}/test_run_{test_run}'
                    if not os.path.exists(data_child_path):
                        os.makedirs(data_child_path)

                    self.__test_runs_dataframes.get(ticker).get(test_run).to_csv(f'{data_child_path}/adj_close.csv')
                    with open(f'{data_child_path}/learning_params.json', 'w') as file:
                        json.dump(self.__test_runs_components_learning_params.get(ticker).get(test_run), file)
                    time_series_components_dataframes = {}
                    for component in self.__test_runs_components.get(ticker).get(test_run).keys():
                        time_series_components_dataframes.update({
                            component: pd.DataFrame(
                                self.__test_runs_components.get(ticker).get(test_run).get(component),
                                index=self.__test_runs_dataframes.get(ticker).get(test_run).index,
                                columns=[component]
                            )
                        })
                        time_series_components_dataframes.get(component).to_csv(f'{data_child_path}/{component}.csv')
                        dump(self.__test_runs_components_scalers.get(ticker).get(test_run).get(component),
                             f'{data_child_path}/{component}_scaler.joblib')
                    time_series_dataframes.update({test_run: time_series_components_dataframes})
                self.__test_runs_components.update({ticker: time_series_dataframes})
                self.__cached_tickers.update({ticker: True})

    def init_datasets(self) -> None:
        for ticker in self.__tickers:
            datasets = {}
            for test_run in self.__test_runs_periods.keys():
                periods = self.__test_runs_periods.get(test_run)
                components_datasets = {}
                for index, component in enumerate(self.__test_runs_components.get(ticker).get(test_run).keys()):
                    training_set_input, training_set_targets = self.df_to_timeseries(
                        self.__test_runs_components.get(ticker).get(test_run).get(component).loc[
                            f'{periods[0].to_date_string()}':f'{periods[1].to_date_string()}'
                        ],
                        self.__test_runs_components_learning_params.get(ticker).get(test_run).get('windows_size')[index]
                    )
                    validation_set_input, validation_set_targets = self.df_to_timeseries(
                        self.__test_runs_components.get(ticker).get(test_run).get(component).loc[
                            f'{periods[1].add(days=1).to_date_string()}':f'{periods[2].to_date_string()}'
                        ],
                        self.__test_runs_components_learning_params.get(ticker).get(test_run).get('windows_size')[index]
                    )
                    test_set_input, test_set_targets = self.df_to_timeseries(
                        self.__test_runs_components.get(ticker).get(test_run).get(component).loc[
                            f'{periods[2].add(days=1).to_date_string()}':f'{periods[3].to_date_string()}'
                        ],
                        self.__test_runs_components_learning_params.get(ticker).get(test_run).get('windows_size')[index]
                    )
                    components_datasets.update({
                        component: {
                            'training': {
                                'inputs': training_set_input,
                                'targets': training_set_targets
                            },
                            'validation': {
                                'inputs': validation_set_input,
                                'targets': validation_set_targets
                            },
                            'test': {
                                'inputs': test_set_input,
                                'targets': test_set_targets
                            }
                        }
                    })
                datasets.update({test_run: components_datasets})
            self.__test_runs_datasets.update({ticker: datasets})

    def init_alternative_dataset(self) -> None:
        for ticker in self.__tickers:
            datasets = {}
            for test_run in self.__test_runs_periods.keys():
                components_datasets = {}
                for component in self.__test_runs_datasets.get(ticker).get(test_run).keys():
                    training_sequences = (self.__test_runs_datasets.get(ticker).get(test_run).get(component)
                                          .get('training').get('inputs'))
                    validation_sequences = (self.__test_runs_datasets.get(ticker).get(test_run).get(component)
                                            .get('validation').get('inputs'))
                    test_sequences = (self.__test_runs_datasets.get(ticker).get(test_run).get(component)
                                      .get('test').get('inputs'))
                    closest_sequences_refs = np.zeros((test_sequences.shape[0],), dtype=int)
                    for index, sequence in enumerate(test_sequences):
                        closest_sequences_refs[index] = self.search_closest_sequence(sequence, training_sequences,
                                                                                     validation_sequences)
                    components_datasets.update({component: closest_sequences_refs})
                datasets.update({test_run: components_datasets})
            self.__test_runs_backtracked_datasets.update({ticker: datasets})

    def reconstruct_and_export_results(self, ticker) -> None:
        for test_run in self.__test_runs_predictions.get(ticker).keys():
            png_path = f'./images/predictions/{ticker}'
            if not os.path.exists(png_path):
                os.makedirs(png_path)

            reconstructed_predictions = self.reconstruct_results(
                ticker, test_run, self.__test_runs_predictions.get(ticker).get(test_run))
            reconstructed_predictions_mae = mean_absolute_error(
                self.__test_runs_dataframes.get(ticker).get(test_run).iloc[-len(reconstructed_predictions):],
                reconstructed_predictions)
            backtracked_predictions_aes = self.calculate_backtrack_aes(
                self.__test_runs_backtracked_predictions.get(ticker).get(test_run))
            self.export_results(ticker, test_run, reconstructed_predictions, reconstructed_predictions_mae,
                                backtracked_predictions_aes)
            self.plot_results(ticker, test_run, reconstructed_predictions, png_path)

    def reconstruct_results(self, ticker, test_run, components_predictions) -> np.ndarray:
        results = [0] * len(components_predictions.get('residue'))
        for component in components_predictions.keys():
            component_scaler = self.__test_runs_components_scalers.get(ticker).get(test_run).get(component)
            component_scaler.inverse_transform(components_predictions.get(component))
            components_predictions.update({
                component: np.array(components_predictions.get(component)).flatten()
            })
        for index in range(len(results)):
            results_index = len(results) - index - 1
            for component in components_predictions.keys():
                component_index = len(components_predictions.get(component)) - index - 1
                results[results_index] += components_predictions.get(component)[component_index]
        return np.array(results).reshape(-1, 1)

    def export_results(self, ticker, test_run, predictions, predictions_mae, backtrack_aes) -> None:
        data_path = f'./results/{ticker}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        model_results = pd.DataFrame({
            'forecasted_values': predictions.flatten(),
            'backtrack_absolute_errors': backtrack_aes.flatten(),
            'test_run_mae': predictions_mae
        }, index=self.__test_runs_dataframes.get(ticker).get(test_run).index[-len(predictions):])
        model_results.to_csv(f'{data_path}/test_run_{test_run}_results.csv', encoding='utf-8',
                             sep=';', decimal=',', index_label='Date')

    def plot_results(self, ticker, test_run, results, path) -> None:
        plt.figure(figsize=(16, 9))
        plt.title(f'{ticker} forecasting results (test run {test_run})')
        plt.plot(self.__test_runs_dataframes.get(ticker).get(test_run), label='adj_close')
        plt.plot(pd.Series(
            results.flatten(), index=self.__test_runs_dataframes.get(ticker).get(test_run).index[-len(results):]),
            label='model_predictions')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'{path}/test_run_{test_run}.png')
        plt.close()
