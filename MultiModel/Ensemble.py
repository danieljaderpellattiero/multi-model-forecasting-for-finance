import os.path
import matplotlib
import numpy as np
import pandas as pd
import pendulum as time
import matplotlib.pyplot as plt

from colorama import Fore, Style
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

matplotlib.use('Agg')


class Ensemble:
    data_path = './data'
    images_path = './images'
    results_path = './results'
    models_predictions_path = './models_predictions'
    ensembling_methods = ['unweighted', 'statically_weighted', 'dynamically_weighted']
    data_plot_color = 'royalblue'
    ensembled_predictions_plot_color = 'magenta'
    models_predictions_plot_colors = ['limegreen', 'coral', 'gold', 'deeppink']

    def __init__(self, enx_data, enx_data_frequency, test_runs, static_ensembling_metric, tickers, models_names
                 ) -> None:
        self.__uuid = 'MM'
        self.__tickers = tickers
        self.__enx_data = enx_data
        self.__test_runs = test_runs
        self.__models_names = models_names
        self.__enx_data_frequency = enx_data_frequency
        self.__static_ensembling_metric = static_ensembling_metric
        self.__data = {}
        self.__models_predictions = {}
        self.__ensembled_predictions = {}

    def run(self) -> None:
        if self.import_dataframes():
            self.merge_predictions()
            self.calculate_ensembled_predictions()
            self.export_predictions_metrics()
            self.plot_predictions('single')
            self.plot_predictions('ensembled')
        else:
            exit(1)

    @staticmethod
    def ensemble_predictions(predictions, static_errors, dynamic_errors, method) -> float:
        if method == 'unweighted':
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
                dataframe_path = (f'{self.data_path}/{ticker}/test_run_{test_run}_adj_close.csv' if not self.__enx_data
                                  else
                                  f'{self.data_path}/{ticker}/test_run_{test_run}_trade_price_'
                                  f'{self.__enx_data_frequency}.csv')
                if os.path.exists(dataframe_path):
                    dataframe = pd.read_csv(dataframe_path, index_col=('Date' if not self.__enx_data else
                                                                       'Trade_timestamp'), parse_dates=True)
                    dataframes.update({test_run: dataframe})
                else:
                    is_data_missing = True
                    print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                          f'Missing test set for {ticker} - Test run {test_run}. {Style.RESET_ALL}')
                models_predictions_dataframes = []
                for model in self.__models_names:
                    if is_data_missing:
                        break
                    model_predictions_path = (f'{self.models_predictions_path}/{model}/{ticker}/'
                                              f'test_run_{test_run}.csv' if not self.__enx_data else
                                              f'{self.models_predictions_path}/{model}/{ticker}/'
                                              f'test_run_{test_run}_{self.__enx_data_frequency}.csv')
                    if os.path.exists(model_predictions_path):
                        dataframe = pd.read_csv(model_predictions_path, encoding='utf-8', sep=';', decimal=',',
                                                index_col=('Date' if not self.__enx_data else 'Trade_timestamp'),
                                                parse_dates=True)
                        dataframe = dataframe.add_prefix(f'{model}_')
                        models_predictions_dataframes.append(dataframe)
                    else:
                        is_data_missing = True
                        print(f'{Fore.LIGHTRED_EX} [ {self.__uuid} ] '
                              f'Missing model predictions for {model} - {ticker} - Test run {test_run}. '
                              f'{Style.RESET_ALL}')
                predictions_dataframes.update({test_run: models_predictions_dataframes})
            self.__data.update({ticker: dataframes})
            self.__models_predictions.update({ticker: predictions_dataframes})
        if not is_data_missing:
            print(f'{Fore.LIGHTGREEN_EX} [ {self.__uuid} ] '
                  f'Data and predictions imported for {len(self.__tickers)} ticker(s). {Style.RESET_ALL}')
        return not is_data_missing

    def merge_predictions(self) -> None:
        for ticker in self.__tickers:
            target_dataframe = {}
            merged_dataframes = {}
            for test_run in range(0, self.__test_runs):
                dataframe = pd.concat(self.__models_predictions.get(ticker).get(test_run), axis=1)
                dataframe.dropna(axis=0, how='any', inplace=True)
                merged_dataframes.update({test_run: dataframe})
                target_dataframe.update({test_run: self.__data.get(ticker).get(test_run).tail(dataframe.shape[0])})
            self.__data.update({ticker: target_dataframe})
            self.__models_predictions.update({ticker: merged_dataframes})

    def plot_predictions(self, mode) -> None:
        for ticker in self.__tickers:
            ticker_png_path = f'{self.images_path}/{ticker}/{mode}_predictions'
            if not os.path.exists(ticker_png_path):
                os.makedirs(ticker_png_path)

            for test_run in range(0, self.__test_runs):
                day = (time.instance(self.__data.get(ticker).get(test_run).index[0])
                       .format('dddd Do [of] MMMM YYYY'))
                title = (f'{ticker} - {day} - test run {test_run} - {mode} predictions' if self.__enx_data else
                         f'{ticker} - test run {test_run} - {mode} predictions')
                plt.figure(figsize=(24, 12))
                plt.title(title)
                plt.plot(self.__data.get(ticker).get(test_run),
                         label=('test_set_adj_close' if not self.__enx_data else 'test_set_trade_price'),
                         color=self.data_plot_color)
                for model_index, model in enumerate(self.__models_names):
                    plt.plot(self.__models_predictions.get(ticker).get(test_run)[f'{model}_forecasted_values'],
                             label=(f'{model}_forecasted_adj_close' if not self.__enx_data else
                                    f'{model}_forecasted_trade_price'),
                             color=self.models_predictions_plot_colors[model_index],
                             alpha=1.0 if mode == 'single' else 0.15)
                if mode == 'ensembled':
                    for method_index, method in enumerate(self.ensembling_methods):
                        ensembled_predictions_plotted, = plt.plot(
                            self.__ensembled_predictions.get(ticker).get(test_run)[method],
                            label=f'{method}', color=self.ensembled_predictions_plot_color, alpha=1.0)
                        plt.legend(loc='best')
                        plt.grid(True)
                        plt.savefig(f'{ticker_png_path}/test_run_{test_run}_{method[0:2]}.png' if not self.__enx_data
                                    else f'{ticker_png_path}/test_run_{test_run}_{method[0:2]}_'
                                         f'{self.__enx_data_frequency}.png')
                        ensembled_predictions_plotted.remove()
                    plt.close()
                else:
                    plt.legend(loc='best')
                    plt.grid(True)
                    plt.savefig(f'{ticker_png_path}/test_run_{test_run}.png' if not self.__enx_data else
                                f'{ticker_png_path}/test_run_{test_run}_{self.__enx_data_frequency}.png')
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
                scaler = MinMaxScaler(copy=True, clip=False)
                self.__data.get(ticker).get(test_run)[:] = scaler.fit_transform(
                    self.__data.get(ticker).get(test_run).to_numpy(copy=True))
                for method in self.ensembling_methods:
                    self.__ensembled_predictions.get(ticker).get(test_run)[method][:] = (
                        scaler.transform(
                            self.__ensembled_predictions.get(ticker).get(test_run)[method].to_numpy(copy=True)
                            .reshape(-1, 1))
                    ).flatten()
                for model in self.__models_names:
                    self.__models_predictions.get(ticker).get(test_run)[f'{model}_forecasted_values'][:] = (
                        scaler.transform(
                            self.__models_predictions.get(ticker).get(test_run)[f'{model}_forecasted_values']
                            .to_numpy(copy=True).reshape(-1, 1))
                    ).flatten()
                for metric in [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error]:
                    metric_dataframe = pd.DataFrame()
                    for method_index, method in enumerate(self.ensembling_methods):
                        metric_dataframe[f'{method[0:2]}_{metric.__name__}'] = pd.Series(
                            metric(self.__data.get(ticker).get(test_run),
                                   self.__ensembled_predictions.get(ticker).get(test_run)[method]))
                    for model in self.__models_names:
                        metric_dataframe[f'{model}_{metric.__name__}'] = pd.Series(
                            metric(self.__data.get(ticker).get(test_run),
                                   self.__models_predictions.get(ticker).get(test_run)[f'{model}_forecasted_values']))
                    metric_dataframe.to_csv(
                        f'{self.results_path}/{ticker}/test_run_{test_run}_{metric.__name__}.csv',
                        index=True, encoding='utf-8', sep=';', decimal=',')
