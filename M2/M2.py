from Config import Config
from DataManager import DataManager
from CMDNLSTM import CMDNLSTM as LSTM


class M2:

    def __init__(self, config_params, tickers) -> None:
        self.__uuid = 'M2'
        self.__alias = 'CEEMDAN-LSTM'
        self.__config = Config(self.__uuid, config_params)
        self.__data_mgmt = DataManager(self.__config, tickers)

    def run(self) -> None:
        self.__data_mgmt.init_periods()
        self.__data_mgmt.import_local_data()
        if not self.__data_mgmt.check_local_data_availability():
            self.__data_mgmt.download_dataframes()
            self.__data_mgmt.decompose_time_series()
            self.__data_mgmt.normalize_time_series_components()
            self.__data_mgmt.init_learning_params()
            self.__data_mgmt.export_time_series_components()
            if not self.__data_mgmt.check_local_data_availability():
                exit(1)
        self.__data_mgmt.init_datasets()
        self.__data_mgmt.init_alternative_dataset()

        for ticker in self.__data_mgmt.tickers:
            predictions = {}
            backtracked_predictions = {}
            for test_run in range(0, self.__config.tr_amt):
                components_predictions = {}
                components_backtracked_predictions = {}
                for index, component in enumerate(self.__data_mgmt.tr_components.get(ticker).get(test_run).keys()):
                    model = LSTM(self.__config, ticker, test_run, component,
                                 self.__data_mgmt.tr_learning_params.get(ticker).get(test_run).get('epochs')[index])
                    if not model.import_model():
                        model.define_model(self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                           .get('training').get('inputs').shape)
                        model.compile_model()
                        model.train_model(self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                          .get('training').get('inputs'),
                                          self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                          .get('training').get('targets'),
                                          self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                          .get('validation').get('inputs'),
                                          self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                          .get('validation').get('targets'))
                        model.evaluate_model(self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                             .get('test').get('inputs'),
                                             self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                             .get('test').get('targets'))

                    components_predictions.update({
                        component: model.predict(self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                                 .get('test').get('inputs'))
                    })

                    backtracked_predictions_tmp = self.__data_mgmt.init_backtrack_buffer(components_predictions
                                                                                         .get(component))
                    for ref_index, ref in enumerate(self.__data_mgmt.tr_bt_datasets.get(ticker).get(test_run)
                                                                                   .get(component)):
                        backtracked_sequence_set = (
                            'training' if ref <= (self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                                  .get('training').get('inputs').shape[0] - 1) else 'validation')
                        actual_ref = (ref if backtracked_sequence_set == 'training'
                                      else ref - self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                      .get('training').get('inputs').shape[0])
                        backtracked_sequence = (self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                                                                            .get(backtracked_sequence_set)
                                                                            .get('inputs')[actual_ref])
                        backtracked_predictions_tmp[ref_index][0] = model.predict(backtracked_sequence
                                                                                  .reshape(1,
                                                                                           self.__config.window_size))
                        backtracked_predictions_tmp[ref_index][1] = (
                            self.__data_mgmt.tr_datasets.get(ticker).get(test_run).get(component)
                            .get(backtracked_sequence_set).get('targets')[actual_ref])
                    components_backtracked_predictions.update({component: backtracked_predictions_tmp})
                predictions.update({test_run: components_predictions})
                backtracked_predictions.update({test_run: components_backtracked_predictions})
            self.__data_mgmt.tr_predictions.update({ticker: predictions})
            self.__data_mgmt.tr_bt_predictions.update({ticker: backtracked_predictions})
            self.__data_mgmt.reconstruct_and_export_results(ticker)
