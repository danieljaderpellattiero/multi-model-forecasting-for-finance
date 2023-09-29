import os.path
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

matplotlib.use('Agg')


class Ensemble:
    data_path = './data'
    png_path = './images'
    results_path = './results'
    models_predictions_path = './models_predictions'
    datasets = ['training', 'validation', 'test']
    ensembling_methods = ['democratic', 'statically_weighted', 'dynamically_weighted']
    data_plot_color = 'royalblue'
    ensembled_predictions_plot_color = 'orchid'
    models_predictions_plot_colors = ['limegreen', 'crimson', 'gold']

    def __init__(self, tickers, test_runs, models_names, static_ensembling_metric) -> None:
        self.__uuid = 'MM'
        self.__tickers = tickers
        self.__test_runs = test_runs
        self.__models_names = models_names
        self.__static_ensembling_metric = static_ensembling_metric
        self.__data = {}
        self.__models_predictions = {}
        self.__ensembled_predictions = {}

    def run(self) -> None:
        if self.import_dataframes():
            self.merge_predictions()
            self.plot_predictions('single')
            self.calculate_ensembled_predictions()
            self.plot_predictions('ensembled')
            self.export_predictions_metrics()
        else:
            exit(1)

    @staticmethod
    def ensemble_predictions(predictions, static_errors, dynamic_errors, method) -> float:
        if method == 'democratic':
            return np.mean(predictions)
        elif method == 'statically_weighted':
            return np.dot(predictions, static_errors) / np.sum(static_errors)
        elif method == 'dynamically_weighted':
            return np.dot(predictions, dynamic_errors) / np.sum(dynamic_errors)

    def import_dataframes(self) -> bool:
        is_data_missing = False
        for ticker in self.__tickers:
            if is_data_missing:
                break
            dataframes = {}
            predictions_dataframes = {}
            for test_run in range(0, self.__test_runs):
                if is_data_missing:
                    break
                sub_dataframes = {}
                for dataset in self.datasets:
                    if is_data_missing:
                        break
                    time_series_path = f'{self.data_path}/{ticker}/test_run_{test_run}_{dataset}.csv'
                    if os.path.exists(time_series_path):
                        sub_dataframes.update({dataset: pd.read_csv(time_series_path, index_col='Date',
                                                                    parse_dates=True)})
                    else:
                        is_data_missing = True
                        print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                              f'Missing data for {ticker} - Test run {test_run} - {dataset} set. {Style.RESET_ALL}')
                dataframes.update({test_run: sub_dataframes})
                sub_predictions_dataframes = []
                for model in self.__models_names:
                    if is_data_missing:
                        break
                    model_predictions_path = (f'{self.models_predictions_path}/{model}/{ticker}/'
                                              f'test_run_{test_run}.csv')
                    if os.path.exists(model_predictions_path):
                        dataframe = pd.read_csv(model_predictions_path, encoding='utf-8', sep=',', decimal='.',
                                                index_col='Date', parse_dates=True)
                        dataframe = dataframe.add_prefix(f'{model}_')
                        sub_predictions_dataframes.append(dataframe)
                    else:
                        is_data_missing = True
                        print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                              f'Missing model predictions for {model} - {ticker} - Test run {test_run}. '
                              f'{Style.RESET_ALL}')
                predictions_dataframes.update({test_run: sub_predictions_dataframes})
            self.__data.update({ticker: dataframes})
            self.__models_predictions.update({ticker: predictions_dataframes})
        if not is_data_missing:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__uuid} ] '
                  f'Data and predictions imported for {len(self.__tickers)} ticker(s). {Style.RESET_ALL}')
        return not is_data_missing

    def merge_predictions(self) -> None:
        for ticker in self.__tickers:
            merged_dataframes = {}
            for test_run in range(0, self.__test_runs):
                dataframe = pd.concat(self.__models_predictions.get(ticker).get(test_run), axis=1)
                dataframe.dropna(axis=0, how='any', inplace=True)
                merged_dataframes.update({test_run: dataframe})
            self.__models_predictions.update({ticker: merged_dataframes})
        pass

    def plot_predictions(self, mode) -> None:
        for ticker in self.__tickers:
            ticker_png_path = f'{self.png_path}/{ticker}/{mode}_predictions'
            if not os.path.exists(ticker_png_path):
                os.makedirs(ticker_png_path)

            for test_run in range(0, self.__test_runs):
                plt.figure(figsize=(16, 9))
                plt.title(f'{ticker} (test run {test_run}) - {mode} predictions')
                plt.plot(self.__data.get(ticker).get(test_run).get('test'), label='test_set_adj_close',
                         color=self.data_plot_color)
                for model_index, model in enumerate(self.__models_names):
                    plt.plot(self.__models_predictions.get(ticker).get(test_run)[f'{model}_forecasted_values'],
                             label=f'{model}_forecasted_adj_close',
                             color=self.models_predictions_plot_colors[model_index])
                if mode == 'ensembled':
                    for method_index, method in enumerate(self.ensembling_methods):
                        ensembled_predictions_plotted, = plt.plot(
                            self.__ensembled_predictions.get(ticker).get(test_run)[method],
                            label=f'{method}', color=self.ensembled_predictions_plot_color)
                        plt.legend(loc='best')
                        plt.grid(True)
                        plt.savefig(f'{ticker_png_path}/test_run_{test_run}_{method[0:2]}.png')
                        ensembled_predictions_plotted.remove()
                    plt.close()
                else:
                    plt.legend(loc='best')
                    plt.grid(True)
                    plt.savefig(f'{ticker_png_path}/test_run_{test_run}.png')
                    plt.close()

    def get_errors(self, ticker, test_run, mode, row_index=-1) -> np.array:
        errors = np.zeros((len(self.__models_names),))
        for model_index, model in enumerate(self.__models_names):
            with np.errstate(divide='raise', invalid='raise'):
                if mode == 'static':
                    errors[model_index] = 1 / self.__models_predictions.get(ticker).get(test_run)[
                        f'{model}_{self.__static_ensembling_metric}'].iloc[0]
                elif mode == 'dynamic':
                    errors[model_index] = 1 / self.__models_predictions.get(ticker).get(test_run)[
                        f'{model}_backtracked_values_aes'].loc[row_index]
        errors = errors / np.sum(errors)
        return errors

    def calculate_ensembled_predictions(self) -> None:
        for ticker in self.__tickers:
            predictions = {}
            for test_run in range(0, self.__test_runs):
                predictions_dataframe = pd.DataFrame(
                    columns=self.ensembling_methods,
                    index=self.__models_predictions.get(ticker).get(test_run).index)
                models_static_errors = self.get_errors(ticker, test_run, 'static')
                for row_index, row in self.__models_predictions.get(ticker).get(test_run).iterrows():
                    models_dynamic_errors = self.get_errors(ticker, test_run, 'dynamic', row_index)
                    models_predictions = np.zeros((len(self.__models_names),))
                    for model_index, model in enumerate(self.__models_names):
                        models_predictions[model_index] = row[f'{model}_forecasted_values']
                    for method_index, method in enumerate(self.ensembling_methods):
                        predictions_dataframe.loc[row_index, method] = self.ensemble_predictions(
                            models_predictions, models_static_errors, models_dynamic_errors, method)
                predictions.update({test_run: predictions_dataframe})
            self.__ensembled_predictions.update({ticker: predictions})

    def export_predictions_metrics(self) -> None:
        for ticker in self.__tickers:
            if not os.path.exists(f'{self.results_path}/{ticker}'):
                os.makedirs(f'{self.results_path}/{ticker}')
            for test_run in range(0, self.__test_runs):
                metrics_dataframe = pd.DataFrame()
                test_set_data = self.__data.get(ticker).get(test_run).get('test').iloc[
                                -self.__ensembled_predictions.get(ticker).get(test_run).shape[0]:]
                for method_index, method in enumerate(self.ensembling_methods):
                    for metric in [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error]:
                        metrics_dataframe[f'{method[0:2]}_{metric.__name__}'] = pd.Series(
                            metric(test_set_data, self.__ensembled_predictions.get(ticker).get(test_run)[method]))
                for model in self.__models_names:
                    for metric in [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error]:
                        metrics_dataframe[f'{model}_{metric.__name__}'] = pd.Series(
                            metric(test_set_data, self.__models_predictions.get(ticker).get(test_run)[
                                f'{model}_forecasted_values']))
                metrics_dataframe.to_csv(f'{self.results_path}/{ticker}/test_run_{test_run}.csv', index=True,
                                         encoding='utf-8', sep=';', decimal='.')
